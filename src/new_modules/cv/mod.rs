//! Computer Vision Module for NumRS2
//!
//! This module provides comprehensive image processing and computer vision
//! operations built on NumRS2's array infrastructure. All operations follow
//! the SCIRS2 integration policy and are implemented in pure Rust.
//!
//! # Module Organization
//!
//! - [`filters`]: Image filtering operations (Gaussian blur, Sobel, Laplacian, bilateral, Canny)
//! - [`morphology`]: Morphological operations (erosion, dilation, opening, closing)
//! - [`features`]: Feature detection (Harris corner, FAST, descriptor extraction, matching)
//! - [`transforms`]: Geometric transforms (resize, rotate, affine, flip, crop, pad)
//!
//! # Image Representation
//!
//! Images are represented using the [`Image`] struct, which wraps NumRS2 arrays:
//! - Grayscale images use a 2D array of shape `[height, width]`
//! - RGB images use a 3D array of shape `[height, width, 3]`
//! - All pixel values are stored as `f64` in the range `[0.0, 1.0]`
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::new_modules::cv::*;
//!
//! // Create a grayscale image from raw data
//! let data = vec![0.0; 64 * 64];
//! let img = Image::from_grayscale(64, 64, &data).expect("valid image");
//!
//! // Apply Gaussian blur
//! let blurred = filters::gaussian_blur(&img, 3, 1.0).expect("blur succeeded");
//! ```
//!
//! # SCIRS2 Integration Policy
//!
//! All modules follow NumRS2's SCIRS2 integration policy:
//! - Array operations: `scirs2_core::ndarray` (NEVER direct ndarray)
//! - Error handling: Based on `NumRs2Error`
//! - Pure Rust implementation (no C/C++ dependencies)

pub mod features;
pub mod filters;
pub mod morphology;
pub mod transforms;

use crate::array::Array;
use crate::error::NumRs2Error;

/// Color space representation for images
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Single-channel grayscale image
    Grayscale,
    /// Three-channel RGB image
    Rgb,
    /// Four-channel RGBA image with alpha
    Rgba,
}

/// Border handling mode for filtering operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderMode {
    /// Pad with a constant value (default: 0.0)
    #[default]
    Constant,
    /// Reflect pixels at the border (e.g., `dcba|abcd|dcba`)
    Reflect,
    /// Replicate the edge pixel value
    Replicate,
    /// Wrap around to the opposite side
    Wrap,
}

/// An image representation wrapping NumRS2 arrays.
///
/// Images store pixel data as `f64` values in the range `[0.0, 1.0]`.
/// Grayscale images have shape `[height, width]`, RGB images have shape
/// `[height, width, 3]`, and RGBA images have shape `[height, width, 4]`.
#[derive(Debug, Clone)]
pub struct Image {
    /// The underlying pixel data
    data: Array<f64>,
    /// Width of the image in pixels
    width: usize,
    /// Height of the image in pixels
    height: usize,
    /// Color space of the image
    color_space: ColorSpace,
}

impl Image {
    /// Creates a new grayscale image from raw pixel data.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - Pixel values in row-major order, expected in `[0.0, 1.0]`
    ///
    /// # Errors
    /// Returns `NumRs2Error::ValueError` if data length does not match width * height
    pub fn from_grayscale(width: usize, height: usize, data: &[f64]) -> Result<Self, NumRs2Error> {
        if data.len() != width * height {
            return Err(NumRs2Error::ValueError(format!(
                "Data length {} does not match image dimensions {}x{} = {}",
                data.len(),
                width,
                height,
                width * height
            )));
        }
        let array = Array::from_vec_shape(data.to_vec(), &[height, width])?;
        Ok(Self {
            data: array,
            width,
            height,
            color_space: ColorSpace::Grayscale,
        })
    }

    /// Creates a new RGB image from raw pixel data.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - Pixel values in row-major order with 3 channels per pixel
    ///
    /// # Errors
    /// Returns `NumRs2Error::ValueError` if data length does not match width * height * 3
    pub fn from_rgb(width: usize, height: usize, data: &[f64]) -> Result<Self, NumRs2Error> {
        if data.len() != width * height * 3 {
            return Err(NumRs2Error::ValueError(format!(
                "Data length {} does not match RGB image dimensions {}x{}x3 = {}",
                data.len(),
                width,
                height,
                width * height * 3
            )));
        }
        let array = Array::from_vec_shape(data.to_vec(), &[height, width, 3])?;
        Ok(Self {
            data: array,
            width,
            height,
            color_space: ColorSpace::Rgb,
        })
    }

    /// Creates a grayscale image filled with zeros (black).
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    pub fn zeros_grayscale(width: usize, height: usize) -> Self {
        Self {
            data: Array::zeros(&[height, width]),
            width,
            height,
            color_space: ColorSpace::Grayscale,
        }
    }

    /// Creates an image from a pre-built Array and color space.
    ///
    /// # Arguments
    /// * `data` - The array containing pixel data
    /// * `color_space` - The color space of the image
    ///
    /// # Errors
    /// Returns `NumRs2Error::ValueError` if the array shape is inconsistent with the color space
    pub fn from_array(data: Array<f64>, color_space: ColorSpace) -> Result<Self, NumRs2Error> {
        let shape = data.shape();
        match color_space {
            ColorSpace::Grayscale => {
                if shape.len() != 2 {
                    return Err(NumRs2Error::ValueError(format!(
                        "Grayscale image requires 2D array, got {}D",
                        shape.len()
                    )));
                }
                Ok(Self {
                    width: shape[1],
                    height: shape[0],
                    data,
                    color_space,
                })
            }
            ColorSpace::Rgb => {
                if shape.len() != 3 || shape[2] != 3 {
                    return Err(NumRs2Error::ValueError(format!(
                        "RGB image requires 3D array with 3 channels, got shape {:?}",
                        shape
                    )));
                }
                Ok(Self {
                    width: shape[1],
                    height: shape[0],
                    data,
                    color_space,
                })
            }
            ColorSpace::Rgba => {
                if shape.len() != 3 || shape[2] != 4 {
                    return Err(NumRs2Error::ValueError(format!(
                        "RGBA image requires 3D array with 4 channels, got shape {:?}",
                        shape
                    )));
                }
                Ok(Self {
                    width: shape[1],
                    height: shape[0],
                    data,
                    color_space,
                })
            }
        }
    }

    /// Returns the width of the image in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the image in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the color space of the image.
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Returns a reference to the underlying array data.
    pub fn data(&self) -> &Array<f64> {
        &self.data
    }

    /// Returns a mutable reference to the underlying array data.
    pub fn data_mut(&mut self) -> &mut Array<f64> {
        &mut self.data
    }

    /// Returns the number of channels in the image.
    pub fn channels(&self) -> usize {
        match self.color_space {
            ColorSpace::Grayscale => 1,
            ColorSpace::Rgb => 3,
            ColorSpace::Rgba => 4,
        }
    }

    /// Gets a pixel value at the specified position.
    ///
    /// For grayscale images, returns a single value.
    /// For RGB images, `channel` must be specified (0=R, 1=G, 2=B).
    ///
    /// # Errors
    /// Returns `NumRs2Error::IndexError` if coordinates are out of bounds
    pub fn get_pixel(&self, row: usize, col: usize, channel: usize) -> Result<f64, NumRs2Error> {
        if row >= self.height || col >= self.width {
            return Err(NumRs2Error::IndexError(format!(
                "Pixel ({}, {}) out of bounds for image {}x{}",
                row, col, self.height, self.width
            )));
        }
        match self.color_space {
            ColorSpace::Grayscale => self.data.get(&[row, col]).map_err(|e| {
                NumRs2Error::IndexError(format!("Failed to access pixel ({}, {}): {}", row, col, e))
            }),
            ColorSpace::Rgb | ColorSpace::Rgba => {
                if channel >= self.channels() {
                    return Err(NumRs2Error::IndexError(format!(
                        "Channel {} out of bounds for {} channels",
                        channel,
                        self.channels()
                    )));
                }
                self.data.get(&[row, col, channel]).map_err(|e| {
                    NumRs2Error::IndexError(format!(
                        "Failed to access pixel ({}, {}, {}): {}",
                        row, col, channel, e
                    ))
                })
            }
        }
    }

    /// Sets a pixel value at the specified position.
    ///
    /// # Errors
    /// Returns `NumRs2Error::IndexError` if coordinates are out of bounds
    pub fn set_pixel(
        &mut self,
        row: usize,
        col: usize,
        channel: usize,
        value: f64,
    ) -> Result<(), NumRs2Error> {
        if row >= self.height || col >= self.width {
            return Err(NumRs2Error::IndexError(format!(
                "Pixel ({}, {}) out of bounds for image {}x{}",
                row, col, self.height, self.width
            )));
        }
        match self.color_space {
            ColorSpace::Grayscale => self.data.set(&[row, col], value).map_err(|e| {
                NumRs2Error::IndexError(format!("Failed to set pixel ({}, {}): {}", row, col, e))
            }),
            ColorSpace::Rgb | ColorSpace::Rgba => {
                if channel >= self.channels() {
                    return Err(NumRs2Error::IndexError(format!(
                        "Channel {} out of bounds for {} channels",
                        channel,
                        self.channels()
                    )));
                }
                self.data.set(&[row, col, channel], value).map_err(|e| {
                    NumRs2Error::IndexError(format!(
                        "Failed to set pixel ({}, {}, {}): {}",
                        row, col, channel, e
                    ))
                })
            }
        }
    }

    /// Converts an RGB image to grayscale using luminance weights.
    ///
    /// Uses the ITU-R BT.709 formula: Y = 0.2126*R + 0.7152*G + 0.0722*B
    ///
    /// # Errors
    /// Returns `NumRs2Error::InvalidOperation` if the image is already grayscale
    pub fn to_grayscale(&self) -> Result<Self, NumRs2Error> {
        match self.color_space {
            ColorSpace::Grayscale => Ok(self.clone()),
            ColorSpace::Rgb | ColorSpace::Rgba => {
                // Built from scratch (every pixel is written exactly once),
                // so a flat row-major Vec + `from_vec_shape` replaces
                // `Array::zeros(..)` + a per-pixel `set()` call: no wasted
                // zero-fill and no per-pixel `Arc::make_mut` unshare check
                // for what can be a multi-megapixel loop.
                let mut gray_vec = Vec::with_capacity(self.height * self.width);
                for row in 0..self.height {
                    for col in 0..self.width {
                        let r = self.data.get(&[row, col, 0]).map_err(|e| {
                            NumRs2Error::ComputationError(format!("Red channel access: {}", e))
                        })?;
                        let g = self.data.get(&[row, col, 1]).map_err(|e| {
                            NumRs2Error::ComputationError(format!("Green channel access: {}", e))
                        })?;
                        let b = self.data.get(&[row, col, 2]).map_err(|e| {
                            NumRs2Error::ComputationError(format!("Blue channel access: {}", e))
                        })?;
                        gray_vec.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    }
                }
                let result = Array::from_vec_shape(gray_vec, &[self.height, self.width])?;
                Ok(Self {
                    data: result,
                    width: self.width,
                    height: self.height,
                    color_space: ColorSpace::Grayscale,
                })
            }
        }
    }

    /// Clamps all pixel values to the range `[0.0, 1.0]`.
    pub fn clamp(&self) -> Self {
        let clamped = self.data.map(|v| v.clamp(0.0, 1.0));
        Self {
            data: clamped,
            width: self.width,
            height: self.height,
            color_space: self.color_space,
        }
    }

    /// Returns the flat pixel data as a vector.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.to_vec()
    }
}

/// Errors specific to computer vision operations
#[derive(Debug, Clone)]
pub enum CvError {
    /// Invalid kernel size (must be odd and positive)
    InvalidKernelSize(usize),
    /// Image dimensions do not match expected format
    DimensionMismatch { expected: String, actual: String },
    /// Parameter out of valid range
    InvalidParameter(String),
    /// Operation requires grayscale image
    RequiresGrayscale,
    /// General computation error
    ComputationFailed(String),
}

impl std::fmt::Display for CvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CvError::InvalidKernelSize(s) => {
                write!(f, "Invalid kernel size: {} (must be odd and positive)", s)
            }
            CvError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            CvError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            CvError::RequiresGrayscale => write!(f, "Operation requires grayscale image"),
            CvError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
        }
    }
}

impl From<CvError> for NumRs2Error {
    fn from(err: CvError) -> Self {
        NumRs2Error::InvalidOperation(err.to_string())
    }
}

// Re-exports for convenience
pub use features::{
    brute_force_match, fast_corner_detect, harris_corner_detect, non_maximum_suppression,
    simple_descriptor, FeatureDescriptor, FeatureMatch, Keypoint,
};
pub use filters::{
    bilateral_filter, box_blur, canny_edge_detect, convolve2d, gaussian_blur, laplacian_filter,
    median_filter, sobel_x, sobel_y,
};
pub use morphology::{
    black_hat, closing, dilation, erosion, morphological_gradient, opening, top_hat,
    StructuringElement,
};
pub use transforms::{
    affine_transform, crop, flip_horizontal, flip_vertical, pad, resize_bilinear, resize_nearest,
    rotate, PadMode,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation_grayscale() {
        let data = vec![0.5; 10 * 10];
        let img = Image::from_grayscale(10, 10, &data);
        assert!(img.is_ok());
        let img = img.expect("test: grayscale image creation should succeed");
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
        assert_eq!(img.channels(), 1);
        assert_eq!(img.color_space(), ColorSpace::Grayscale);
    }

    #[test]
    fn test_image_creation_rgb() {
        let data = vec![0.5; 10 * 10 * 3];
        let img = Image::from_rgb(10, 10, &data);
        assert!(img.is_ok());
        let img = img.expect("test: RGB image creation should succeed");
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.color_space(), ColorSpace::Rgb);
    }

    #[test]
    fn test_image_creation_invalid_size() {
        let data = vec![0.5; 50];
        let result = Image::from_grayscale(10, 10, &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pixel_access() {
        let mut data = vec![0.0; 4 * 4];
        data[0] = 1.0;
        data[5] = 0.5; // row=1, col=1
        let img = Image::from_grayscale(4, 4, &data).expect("test: image creation should succeed");
        let v = img
            .get_pixel(0, 0, 0)
            .expect("test: get_pixel should succeed");
        assert!((v - 1.0).abs() < 1e-10);
        let v2 = img
            .get_pixel(1, 1, 0)
            .expect("test: get_pixel should succeed");
        assert!((v2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_set_pixel() {
        let mut img = Image::zeros_grayscale(4, 4);
        img.set_pixel(1, 2, 0, 0.75)
            .expect("test: set_pixel should succeed");
        let v = img
            .get_pixel(1, 2, 0)
            .expect("test: get_pixel after set should succeed");
        assert!((v - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_to_grayscale() {
        // White pixel: R=1, G=1, B=1 -> gray should be ~1.0
        let data = vec![1.0; 2 * 2 * 3];
        let img = Image::from_rgb(2, 2, &data).expect("test: RGB image creation should succeed");
        let gray = img
            .to_grayscale()
            .expect("test: to_grayscale should succeed");
        assert_eq!(gray.color_space(), ColorSpace::Grayscale);
        let v = gray
            .get_pixel(0, 0, 0)
            .expect("test: get grayscale pixel should succeed");
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clamp() {
        let data = vec![-0.5, 0.5, 1.5, 0.0];
        let img = Image::from_grayscale(2, 2, &data).expect("test: image creation should succeed");
        let clamped = img.clamp();
        let result = clamped.to_vec();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zeros_grayscale() {
        let img = Image::zeros_grayscale(8, 8);
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
        for row in 0..8 {
            for col in 0..8 {
                let v = img
                    .get_pixel(row, col, 0)
                    .expect("test: get_pixel should succeed");
                assert!((v).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pixel_out_of_bounds() {
        let img = Image::zeros_grayscale(4, 4);
        assert!(img.get_pixel(5, 0, 0).is_err());
        assert!(img.get_pixel(0, 5, 0).is_err());
    }

    #[test]
    fn test_from_array() {
        let arr = Array::zeros(&[10, 10]);
        let img = Image::from_array(arr, ColorSpace::Grayscale);
        assert!(img.is_ok());
        let img = img.expect("test: from_array should succeed");
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
    }
}

//! Geometric Image Transformations
//!
//! Provides geometric transformation operations for images including
//! resizing, rotation, flipping, cropping, padding, and general affine
//! transformations.
//!
//! # Operations
//!
//! - **Resize**: Nearest neighbor and bilinear interpolation
//! - **Rotate**: Arbitrary angle rotation around image center
//! - **Flip**: Horizontal and vertical flipping
//! - **Crop**: Extract rectangular subregions
//! - **Pad**: Add borders with various fill modes
//! - **Affine transform**: General 2x3 affine matrix transformation
//!
//! # Interpolation
//!
//! Two interpolation methods are supported:
//! - **Nearest neighbor**: Fast, no blurring, produces blocky results
//! - **Bilinear**: Smooth interpolation between 4 neighboring pixels
//!
//! # SCIRS2 Policy
//!
//! All implementations use `crate::array::Array` for data and follow
//! the pure Rust requirement.

use super::{ColorSpace, CvError, Image};
use crate::error::NumRs2Error;

/// Padding modes for the `pad` function.
///
/// Determines how pixel values are generated in the padded border region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Fill padded region with a constant value (0.0)
    #[default]
    Constant,
    /// Reflect pixels at the border (e.g., `dcba|abcd|dcba`)
    Reflect,
    /// Replicate the nearest edge pixel value
    Replicate,
    /// Wrap around to the opposite side of the image
    Wrap,
}

/// Resizes an image using nearest neighbor interpolation.
///
/// Each pixel in the output image maps to the nearest pixel in the input
/// image. This method is fast but produces blocky artifacts, especially
/// when upscaling.
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `new_width` - Target width in pixels
/// * `new_height` - Target height in pixels
///
/// # Returns
/// A new image with the specified dimensions
///
/// # Errors
/// Returns error if new dimensions are zero
pub fn resize_nearest(
    img: &Image,
    new_width: usize,
    new_height: usize,
) -> Result<Image, NumRs2Error> {
    if new_width == 0 || new_height == 0 {
        return Err(CvError::InvalidParameter(
            "Target dimensions must be greater than zero".to_string(),
        )
        .into());
    }

    let old_w = img.width();
    let old_h = img.height();
    let channels = img.channels();

    let mut result = create_output_image(new_width, new_height, img.color_space())?;

    let x_ratio = old_w as f64 / new_width as f64;
    let y_ratio = old_h as f64 / new_height as f64;

    for row in 0..new_height {
        for col in 0..new_width {
            // Map output pixel center to input coordinates
            let src_row = ((row as f64 + 0.5) * y_ratio - 0.5)
                .round()
                .max(0.0)
                .min((old_h - 1) as f64) as usize;
            let src_col = ((col as f64 + 0.5) * x_ratio - 0.5)
                .round()
                .max(0.0)
                .min((old_w - 1) as f64) as usize;

            for ch in 0..channels {
                let val = img.get_pixel(src_row, src_col, ch).unwrap_or(0.0);
                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "resize_nearest set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Resizes an image using bilinear interpolation.
///
/// Each pixel in the output image is computed by bilinear interpolation
/// of the 4 nearest pixels in the input image. This produces smoother
/// results than nearest neighbor but is computationally more expensive.
///
/// The interpolation formula for a point `(x, y)` between pixels is:
/// ```text
/// f(x, y) = (1-dx)(1-dy)*f(x0,y0) + dx*(1-dy)*f(x1,y0)
///          + (1-dx)*dy*f(x0,y1) + dx*dy*f(x1,y1)
/// ```
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `new_width` - Target width in pixels
/// * `new_height` - Target height in pixels
///
/// # Returns
/// A new image with the specified dimensions
///
/// # Errors
/// Returns error if new dimensions are zero
pub fn resize_bilinear(
    img: &Image,
    new_width: usize,
    new_height: usize,
) -> Result<Image, NumRs2Error> {
    if new_width == 0 || new_height == 0 {
        return Err(CvError::InvalidParameter(
            "Target dimensions must be greater than zero".to_string(),
        )
        .into());
    }

    let old_w = img.width();
    let old_h = img.height();
    let channels = img.channels();

    let mut result = create_output_image(new_width, new_height, img.color_space())?;

    let x_ratio = if new_width > 1 {
        (old_w as f64 - 1.0) / (new_width as f64 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if new_height > 1 {
        (old_h as f64 - 1.0) / (new_height as f64 - 1.0)
    } else {
        0.0
    };

    for row in 0..new_height {
        for col in 0..new_width {
            let src_y = row as f64 * y_ratio;
            let src_x = col as f64 * x_ratio;

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(old_h - 1);
            let x1 = (x0 + 1).min(old_w - 1);

            let dy = src_y - y0 as f64;
            let dx = src_x - x0 as f64;

            for ch in 0..channels {
                let p00 = img.get_pixel(y0, x0, ch).unwrap_or(0.0);
                let p01 = img.get_pixel(y0, x1, ch).unwrap_or(0.0);
                let p10 = img.get_pixel(y1, x0, ch).unwrap_or(0.0);
                let p11 = img.get_pixel(y1, x1, ch).unwrap_or(0.0);

                // Bilinear interpolation
                let val = p00 * (1.0 - dx) * (1.0 - dy)
                    + p01 * dx * (1.0 - dy)
                    + p10 * (1.0 - dx) * dy
                    + p11 * dx * dy;

                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "resize_bilinear set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Rotates an image by the specified angle around its center.
///
/// The rotation is performed by computing the inverse mapping from output
/// to input coordinates and using bilinear interpolation. Pixels that
/// map outside the original image boundaries are filled with black (0.0).
///
/// The output image has the same dimensions as the input. Content near
/// corners may be clipped for large rotation angles.
///
/// The rotation uses the standard rotation matrix:
/// ```text
/// [cos(theta)  -sin(theta)]
/// [sin(theta)   cos(theta)]
/// ```
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `angle_degrees` - Rotation angle in degrees (positive = counter-clockwise)
///
/// # Returns
/// A new rotated image with the same dimensions
///
/// # Errors
/// Returns error on computation failure
pub fn rotate(img: &Image, angle_degrees: f64) -> Result<Image, NumRs2Error> {
    let h = img.height();
    let w = img.width();
    let channels = img.channels();

    let mut result = create_output_image(w, h, img.color_space())?;

    // Convert angle to radians
    let angle_rad = angle_degrees * std::f64::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Center of the image
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    for row in 0..h {
        for col in 0..w {
            // Translate to origin, apply inverse rotation, translate back
            let dx = col as f64 - cx;
            let dy = row as f64 - cy;

            // Inverse rotation (rotate by -angle to find source pixel)
            let src_x = dx * cos_a + dy * sin_a + cx;
            let src_y = -dx * sin_a + dy * cos_a + cy;

            // Check if source is within bounds and interpolate bilinearly
            if src_x >= 0.0 && src_x < (w - 1) as f64 && src_y >= 0.0 && src_y < (h - 1) as f64 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);

                let dx_frac = src_x - x0 as f64;
                let dy_frac = src_y - y0 as f64;

                for ch in 0..channels {
                    let p00 = img.get_pixel(y0, x0, ch).unwrap_or(0.0);
                    let p01 = img.get_pixel(y0, x1, ch).unwrap_or(0.0);
                    let p10 = img.get_pixel(y1, x0, ch).unwrap_or(0.0);
                    let p11 = img.get_pixel(y1, x1, ch).unwrap_or(0.0);

                    let val = p00 * (1.0 - dx_frac) * (1.0 - dy_frac)
                        + p01 * dx_frac * (1.0 - dy_frac)
                        + p10 * (1.0 - dx_frac) * dy_frac
                        + p11 * dx_frac * dy_frac;

                    result.set_pixel(row, col, ch, val).map_err(|e| {
                        NumRs2Error::ComputationError(format!(
                            "rotate set pixel ({}, {}, {}): {}",
                            row, col, ch, e
                        ))
                    })?;
                }
            }
            // Pixels outside bounds remain 0.0 (black)
        }
    }

    Ok(result)
}

/// Flips an image horizontally (left-right mirror).
///
/// Each row of pixels is reversed so that the leftmost pixel becomes
/// the rightmost and vice versa.
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
///
/// # Returns
/// A new horizontally-flipped image
///
/// # Errors
/// Returns error on computation failure
pub fn flip_horizontal(img: &Image) -> Result<Image, NumRs2Error> {
    let h = img.height();
    let w = img.width();
    let channels = img.channels();

    let mut result = create_output_image(w, h, img.color_space())?;

    for row in 0..h {
        for col in 0..w {
            let src_col = w - 1 - col;
            for ch in 0..channels {
                let val = img.get_pixel(row, src_col, ch).unwrap_or(0.0);
                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "flip_horizontal set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Flips an image vertically (top-bottom mirror).
///
/// The order of rows is reversed so that the top row becomes the bottom
/// row and vice versa.
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
///
/// # Returns
/// A new vertically-flipped image
///
/// # Errors
/// Returns error on computation failure
pub fn flip_vertical(img: &Image) -> Result<Image, NumRs2Error> {
    let h = img.height();
    let w = img.width();
    let channels = img.channels();

    let mut result = create_output_image(w, h, img.color_space())?;

    for row in 0..h {
        let src_row = h - 1 - row;
        for col in 0..w {
            for ch in 0..channels {
                let val = img.get_pixel(src_row, col, ch).unwrap_or(0.0);
                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "flip_vertical set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Crops a rectangular region from an image.
///
/// Extracts pixels from the rectangle defined by the top-left corner
/// `(x, y)` with the given `width` and `height`. The coordinate system
/// uses `x` for column offset and `y` for row offset.
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `x` - Left column of the crop region (column offset)
/// * `y` - Top row of the crop region (row offset)
/// * `width` - Width of the crop region in pixels
/// * `height` - Height of the crop region in pixels
///
/// # Returns
/// A new image containing the cropped region
///
/// # Errors
/// Returns error if the crop region extends beyond the image boundaries,
/// or if width/height are zero
pub fn crop(
    img: &Image,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<Image, NumRs2Error> {
    if width == 0 || height == 0 {
        return Err(CvError::InvalidParameter(
            "Crop dimensions must be greater than zero".to_string(),
        )
        .into());
    }
    if x + width > img.width() || y + height > img.height() {
        return Err(CvError::InvalidParameter(format!(
            "Crop region (x={}, y={}, w={}, h={}) exceeds image bounds ({}x{})",
            x,
            y,
            width,
            height,
            img.width(),
            img.height()
        ))
        .into());
    }

    let channels = img.channels();
    let mut result = create_output_image(width, height, img.color_space())?;

    for row in 0..height {
        for col in 0..width {
            let src_row = y + row;
            let src_col = x + col;
            for ch in 0..channels {
                let val = img.get_pixel(src_row, src_col, ch).unwrap_or(0.0);
                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "crop set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Pads an image by adding borders on all four sides.
///
/// The border pixels are filled according to the specified `PadMode`:
///
/// - **Constant**: Fill with 0.0 (black)
/// - **Reflect**: Mirror pixel values at the edge (`dcba|abcd|dcba`)
/// - **Replicate**: Extend the nearest edge pixel outward
/// - **Wrap**: Tile the image cyclically
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `top` - Number of rows to add at the top
/// * `bottom` - Number of rows to add at the bottom
/// * `left` - Number of columns to add on the left
/// * `right` - Number of columns to add on the right
/// * `mode` - Padding mode
///
/// # Returns
/// A new padded image
///
/// # Errors
/// Returns error on computation failure
pub fn pad(
    img: &Image,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: PadMode,
) -> Result<Image, NumRs2Error> {
    let old_h = img.height();
    let old_w = img.width();
    let new_h = old_h + top + bottom;
    let new_w = old_w + left + right;
    let channels = img.channels();

    if new_h == 0 || new_w == 0 {
        return Err(CvError::InvalidParameter(
            "Padded image dimensions must be greater than zero".to_string(),
        )
        .into());
    }

    let mut result = create_output_image(new_w, new_h, img.color_space())?;

    for row in 0..new_h {
        for col in 0..new_w {
            // Map padded coordinate to source coordinate
            let src_row = map_pad_coordinate(row, top, old_h, mode);
            let src_col = map_pad_coordinate(col, left, old_w, mode);

            for ch in 0..channels {
                let val = match (src_row, src_col) {
                    (Some(sr), Some(sc)) => img.get_pixel(sr, sc, ch).unwrap_or(0.0),
                    _ => 0.0, // Constant mode for out-of-range
                };
                result.set_pixel(row, col, ch, val).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "pad set pixel ({}, {}, {}): {}",
                        row, col, ch, e
                    ))
                })?;
            }
        }
    }

    Ok(result)
}

/// Maps a padded output coordinate back to a source image coordinate.
///
/// Returns `Some(index)` for valid source coordinates, or `None` for
/// positions that should be filled with the constant value.
fn map_pad_coordinate(
    out_idx: usize,
    pad_before: usize,
    src_size: usize,
    mode: PadMode,
) -> Option<usize> {
    if src_size == 0 {
        return None;
    }

    let signed_idx = out_idx as isize - pad_before as isize;

    // Within original image bounds -- all modes agree
    if signed_idx >= 0 && (signed_idx as usize) < src_size {
        return Some(signed_idx as usize);
    }

    match mode {
        PadMode::Constant => None,
        PadMode::Reflect => Some(reflect_coordinate(signed_idx, src_size as isize)),
        PadMode::Replicate => {
            if signed_idx < 0 {
                Some(0)
            } else {
                Some(src_size - 1)
            }
        }
        PadMode::Wrap => {
            let wrapped =
                ((signed_idx % src_size as isize) + src_size as isize) % src_size as isize;
            Some(wrapped as usize)
        }
    }
}

/// Reflects a coordinate at the boundaries using `dcba|abcd|dcba` pattern.
///
/// For an array of length `len`, indices outside `[0, len)` are reflected:
/// - Index -1 maps to 0 (mirror at left edge)
/// - Index -2 maps to 1
/// - Index `len` maps to `len-2` (mirror at right edge)
///
/// For repeated reflections (indices far outside), the pattern repeats
/// periodically.
fn reflect_coordinate(idx: isize, len: isize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    // Normalize to positive range using modular arithmetic
    let normalized = ((idx % period) + period) % period;
    if normalized < len {
        normalized as usize
    } else {
        (period - normalized) as usize
    }
}

/// Applies a 2x3 affine transformation to an image.
///
/// The affine transformation maps output coordinates `(col, row)` to input
/// coordinates `(src_col, src_row)` using:
///
/// ```text
/// src_col = a00 * col + a01 * row + a02
/// src_row = a10 * col + a11 * row + a12
/// ```
///
/// The matrix parameter is provided as:
/// - `matrix[0]` = `[a00, a01, a02]` (column/x mapping)
/// - `matrix[1]` = `[a10, a11, a12]` (row/y mapping)
///
/// This uses the **inverse mapping** approach: for each output pixel, the
/// matrix is applied to compute the corresponding input coordinate, and
/// bilinear interpolation is used to sample the source image. Pixels mapping
/// outside the input bounds are set to 0.0 (black).
///
/// The output image has the same dimensions as the input.
///
/// Supports grayscale, RGB, and RGBA images.
///
/// # Arguments
/// * `img` - Input image
/// * `matrix` - 2x3 affine transformation matrix (inverse mapping)
///
/// # Returns
/// A new image with the affine transformation applied
///
/// # Errors
/// Returns error on computation failure
pub fn affine_transform(img: &Image, matrix: &[[f64; 3]; 2]) -> Result<Image, NumRs2Error> {
    let h = img.height();
    let w = img.width();
    let channels = img.channels();

    let mut result = create_output_image(w, h, img.color_space())?;

    let a00 = matrix[0][0];
    let a01 = matrix[0][1];
    let a02 = matrix[0][2];
    let a10 = matrix[1][0];
    let a11 = matrix[1][1];
    let a12 = matrix[1][2];

    for row in 0..h {
        for col in 0..w {
            // Compute source coordinates via the inverse mapping
            let src_x = a00 * col as f64 + a01 * row as f64 + a02;
            let src_y = a10 * col as f64 + a11 * row as f64 + a12;

            // Bilinear interpolation within bounds
            if src_x >= 0.0 && src_x < (w - 1) as f64 && src_y >= 0.0 && src_y < (h - 1) as f64 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);

                let dx = src_x - x0 as f64;
                let dy = src_y - y0 as f64;

                for ch in 0..channels {
                    let p00 = img.get_pixel(y0, x0, ch).unwrap_or(0.0);
                    let p01 = img.get_pixel(y0, x1, ch).unwrap_or(0.0);
                    let p10 = img.get_pixel(y1, x0, ch).unwrap_or(0.0);
                    let p11 = img.get_pixel(y1, x1, ch).unwrap_or(0.0);

                    let val = p00 * (1.0 - dx) * (1.0 - dy)
                        + p01 * dx * (1.0 - dy)
                        + p10 * (1.0 - dx) * dy
                        + p11 * dx * dy;

                    result.set_pixel(row, col, ch, val).map_err(|e| {
                        NumRs2Error::ComputationError(format!(
                            "affine_transform set pixel ({}, {}, {}): {}",
                            row, col, ch, e
                        ))
                    })?;
                }
            }
            // Pixels outside bounds remain 0.0 (black)
        }
    }

    Ok(result)
}

/// Creates an output image of the given dimensions and color space.
///
/// This helper centralizes the construction of empty output images for
/// all transform operations, supporting grayscale, RGB, and RGBA.
fn create_output_image(
    width: usize,
    height: usize,
    color_space: ColorSpace,
) -> Result<Image, NumRs2Error> {
    match color_space {
        ColorSpace::Grayscale => Ok(Image::zeros_grayscale(width, height)),
        ColorSpace::Rgb => {
            let data = vec![0.0; width * height * 3];
            Image::from_rgb(width, height, &data)
        }
        ColorSpace::Rgba => {
            let data = vec![0.0; width * height * 4];
            let arr = crate::array::Array::from_vec(data).reshape(&[height, width, 4]);
            Image::from_array(arr, ColorSpace::Rgba)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a small grayscale test image with a linear gradient of values.
    fn make_test_image(w: usize, h: usize) -> Image {
        let mut data = vec![0.0; w * h];
        for row in 0..h {
            for col in 0..w {
                data[row * w + col] = (row * w + col) as f64 / (w * h) as f64;
            }
        }
        Image::from_grayscale(w, h, &data).expect("test: image creation should succeed")
    }

    /// Creates a test image with a bright square in the center.
    fn make_centered_square(size: usize) -> Image {
        let mut data = vec![0.0; size * size];
        let quarter = size / 4;
        for row in quarter..(size - quarter) {
            for col in quarter..(size - quarter) {
                data[row * size + col] = 1.0;
            }
        }
        Image::from_grayscale(size, size, &data).expect("test: image creation should succeed")
    }

    // ========================================================================
    // resize_nearest tests
    // ========================================================================

    #[test]
    fn test_resize_nearest_upscale() {
        let img = make_test_image(4, 4);
        let resized =
            resize_nearest(&img, 8, 8).expect("test: nearest resize upscale should succeed");
        assert_eq!(resized.width(), 8);
        assert_eq!(resized.height(), 8);
        // All values should be in valid range
        for row in 0..8 {
            for col in 0..8 {
                let val = resized
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Pixel at ({},{}) out of range: {}",
                    row,
                    col,
                    val
                );
            }
        }
    }

    #[test]
    fn test_resize_nearest_downscale() {
        let img = make_test_image(8, 8);
        let resized = resize_nearest(&img, 4, 4).expect("test: nearest downscale should succeed");
        assert_eq!(resized.width(), 4);
        assert_eq!(resized.height(), 4);
    }

    #[test]
    fn test_resize_nearest_invalid_dimensions() {
        let img = make_test_image(4, 4);
        assert!(
            resize_nearest(&img, 0, 4).is_err(),
            "Should reject zero width"
        );
        assert!(
            resize_nearest(&img, 4, 0).is_err(),
            "Should reject zero height"
        );
    }

    // ========================================================================
    // resize_bilinear tests
    // ========================================================================

    #[test]
    fn test_resize_bilinear_upscale() {
        let img = make_test_image(4, 4);
        let resized =
            resize_bilinear(&img, 8, 8).expect("test: bilinear resize upscale should succeed");
        assert_eq!(resized.width(), 8);
        assert_eq!(resized.height(), 8);
    }

    #[test]
    fn test_resize_bilinear_preserves_corners() {
        // Corner pixels should be exactly preserved in bilinear resize
        let img = make_test_image(4, 4);
        let resized = resize_bilinear(&img, 8, 8).expect("test: bilinear resize should succeed");

        let orig_tl = img
            .get_pixel(0, 0, 0)
            .expect("test: pixel access should succeed");
        let resized_tl = resized
            .get_pixel(0, 0, 0)
            .expect("test: pixel access should succeed");
        assert!(
            (orig_tl - resized_tl).abs() < 1e-6,
            "Top-left corner should be preserved: orig={}, resized={}",
            orig_tl,
            resized_tl
        );

        let orig_br = img
            .get_pixel(3, 3, 0)
            .expect("test: pixel access should succeed");
        let resized_br = resized
            .get_pixel(7, 7, 0)
            .expect("test: pixel access should succeed");
        assert!(
            (orig_br - resized_br).abs() < 1e-6,
            "Bottom-right corner should be preserved: orig={}, resized={}",
            orig_br,
            resized_br
        );
    }

    #[test]
    fn test_resize_bilinear_constant_image() {
        let data = vec![0.5; 4 * 4];
        let img = Image::from_grayscale(4, 4, &data).expect("test: image creation should succeed");
        let resized = resize_bilinear(&img, 8, 8).expect("test: bilinear resize should succeed");
        for row in 0..8 {
            for col in 0..8 {
                let v = resized
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (v - 0.5).abs() < 1e-6,
                    "Constant image should remain constant at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    // ========================================================================
    // rotate tests
    // ========================================================================

    #[test]
    fn test_rotate_zero_degrees() {
        let img = make_test_image(8, 8);
        let rotated = rotate(&img, 0.0).expect("test: zero rotation should succeed");
        assert_eq!(rotated.width(), 8);
        assert_eq!(rotated.height(), 8);

        // Interior pixels should be nearly identical
        for row in 1..7 {
            for col in 1..7 {
                let orig = img
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                let rot = rotated
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (orig - rot).abs() < 1e-6,
                    "Zero rotation at ({}, {}): orig={}, rot={}",
                    row,
                    col,
                    orig,
                    rot
                );
            }
        }
    }

    #[test]
    fn test_rotate_180_degrees() {
        let img = make_test_image(8, 8);
        let rotated = rotate(&img, 180.0).expect("test: 180-degree rotation should succeed");

        // After 180-degree rotation, pixel at (r, c) should approximate (h-1-r, w-1-c)
        for row in 2..6 {
            for col in 2..6 {
                let orig = img
                    .get_pixel(7 - row, 7 - col, 0)
                    .expect("test: pixel access should succeed");
                let rot = rotated
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (orig - rot).abs() < 0.05,
                    "180-deg rotation at ({}, {}): expected ~{}, got {}",
                    row,
                    col,
                    orig,
                    rot
                );
            }
        }
    }

    #[test]
    fn test_rotate_symmetric_image_90() {
        let img = make_centered_square(16);
        let rotated = rotate(&img, 90.0).expect("test: 90-degree rotation should succeed");
        // Center pixel of a symmetric square should still be bright
        let center = rotated
            .get_pixel(8, 8, 0)
            .expect("test: pixel access should succeed");
        assert!(
            center > 0.5,
            "Center of symmetric image should remain bright after 90-degree rotation: got {}",
            center
        );
    }

    // ========================================================================
    // flip tests
    // ========================================================================

    #[test]
    fn test_flip_horizontal_pixel_values() {
        let img = make_test_image(8, 8);
        let flipped = flip_horizontal(&img).expect("test: horizontal flip should succeed");
        assert_eq!(flipped.width(), 8);
        assert_eq!(flipped.height(), 8);

        for row in 0..8 {
            for col in 0..8 {
                let orig = img
                    .get_pixel(row, 7 - col, 0)
                    .expect("test: pixel access should succeed");
                let flip = flipped
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (orig - flip).abs() < 1e-10,
                    "Horizontal flip at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    flip
                );
            }
        }
    }

    #[test]
    fn test_flip_horizontal_double_is_identity() {
        let img = make_test_image(8, 8);
        let flipped = flip_horizontal(&img).expect("test: first flip");
        let double_flipped = flip_horizontal(&flipped).expect("test: second flip");

        for row in 0..8 {
            for col in 0..8 {
                let orig = img.get_pixel(row, col, 0).expect("test: orig pixel");
                let df = double_flipped
                    .get_pixel(row, col, 0)
                    .expect("test: df pixel");
                assert!(
                    (orig - df).abs() < 1e-10,
                    "Double horizontal flip should be identity at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_flip_vertical_pixel_values() {
        let img = make_test_image(8, 8);
        let flipped = flip_vertical(&img).expect("test: vertical flip should succeed");

        for row in 0..8 {
            for col in 0..8 {
                let orig = img
                    .get_pixel(7 - row, col, 0)
                    .expect("test: pixel access should succeed");
                let flip = flipped
                    .get_pixel(row, col, 0)
                    .expect("test: pixel access should succeed");
                assert!(
                    (orig - flip).abs() < 1e-10,
                    "Vertical flip at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    flip
                );
            }
        }
    }

    #[test]
    fn test_flip_vertical_double_is_identity() {
        let img = make_test_image(8, 8);
        let flipped = flip_vertical(&img).expect("test: first flip");
        let double_flipped = flip_vertical(&flipped).expect("test: second flip");

        for row in 0..8 {
            for col in 0..8 {
                let orig = img.get_pixel(row, col, 0).expect("test: orig pixel");
                let df = double_flipped
                    .get_pixel(row, col, 0)
                    .expect("test: df pixel");
                assert!(
                    (orig - df).abs() < 1e-10,
                    "Double vertical flip should be identity at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    // ========================================================================
    // crop tests
    // ========================================================================

    #[test]
    fn test_crop_basic() {
        let img = make_test_image(8, 8);
        let cropped = crop(&img, 2, 2, 4, 4).expect("test: crop should succeed");
        assert_eq!(cropped.width(), 4);
        assert_eq!(cropped.height(), 4);

        for row in 0..4 {
            for col in 0..4 {
                let orig = img
                    .get_pixel(row + 2, col + 2, 0)
                    .expect("test: orig pixel");
                let cr = cropped.get_pixel(row, col, 0).expect("test: cropped pixel");
                assert!(
                    (orig - cr).abs() < 1e-10,
                    "Cropped pixel at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    cr
                );
            }
        }
    }

    #[test]
    fn test_crop_full_image() {
        let img = make_test_image(8, 8);
        let cropped = crop(&img, 0, 0, 8, 8).expect("test: full crop should succeed");
        assert_eq!(cropped.width(), 8);
        assert_eq!(cropped.height(), 8);
    }

    #[test]
    fn test_crop_out_of_bounds() {
        let img = make_test_image(8, 8);
        assert!(
            crop(&img, 6, 6, 4, 4).is_err(),
            "Should fail when region exceeds bounds"
        );
    }

    #[test]
    fn test_crop_zero_dimensions() {
        let img = make_test_image(8, 8);
        assert!(
            crop(&img, 0, 0, 0, 4).is_err(),
            "Should fail with zero width"
        );
        assert!(
            crop(&img, 0, 0, 4, 0).is_err(),
            "Should fail with zero height"
        );
    }

    // ========================================================================
    // pad tests
    // ========================================================================

    #[test]
    fn test_pad_constant_values() {
        let img = make_test_image(4, 4);
        let padded =
            pad(&img, 2, 2, 2, 2, PadMode::Constant).expect("test: constant pad should succeed");
        assert_eq!(padded.width(), 8);
        assert_eq!(padded.height(), 8);

        // Border should be 0.0
        let corner = padded.get_pixel(0, 0, 0).expect("test: corner pixel");
        assert!(
            corner.abs() < 1e-10,
            "Constant pad border should be 0.0: got {}",
            corner
        );

        // Interior should match original
        for row in 0..4 {
            for col in 0..4 {
                let orig = img.get_pixel(row, col, 0).expect("test: orig pixel");
                let padded_val = padded
                    .get_pixel(row + 2, col + 2, 0)
                    .expect("test: padded pixel");
                assert!(
                    (orig - padded_val).abs() < 1e-10,
                    "Padded interior at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    padded_val
                );
            }
        }
    }

    #[test]
    fn test_pad_replicate() {
        let data = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        ];
        let img = Image::from_grayscale(4, 4, &data).expect("test: image creation should succeed");
        let padded =
            pad(&img, 1, 1, 1, 1, PadMode::Replicate).expect("test: replicate pad should succeed");
        assert_eq!(padded.width(), 6);
        assert_eq!(padded.height(), 6);

        // Top-left corner should replicate img[0,0] = 0.1
        let corner = padded.get_pixel(0, 0, 0).expect("test: corner pixel");
        assert!(
            (corner - 0.1).abs() < 1e-10,
            "Replicate pad top-left: expected 0.1, got {}",
            corner
        );

        // Bottom-right corner should replicate img[3,3] = 0.5
        let br = padded.get_pixel(5, 5, 0).expect("test: bottom-right pixel");
        assert!(
            (br - 0.5).abs() < 1e-10,
            "Replicate pad bottom-right: expected 0.5, got {}",
            br
        );
    }

    #[test]
    fn test_pad_wrap() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image::from_grayscale(2, 2, &data).expect("test: image creation should succeed");
        let padded = pad(&img, 1, 1, 1, 1, PadMode::Wrap).expect("test: wrap pad should succeed");
        assert_eq!(padded.width(), 4);
        assert_eq!(padded.height(), 4);

        // Position (0,0) in padded = source at (-1,-1) => wraps to (1,1) = 4.0
        let wrapped = padded.get_pixel(0, 0, 0).expect("test: wrapped pixel");
        assert!(
            (wrapped - 4.0).abs() < 1e-10,
            "Wrap pad (-1,-1) should be 4.0: got {}",
            wrapped
        );
    }

    #[test]
    fn test_pad_reflect_interior_preserved() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let img = Image::from_grayscale(3, 3, &data).expect("test: image creation should succeed");
        let padded =
            pad(&img, 1, 1, 1, 1, PadMode::Reflect).expect("test: reflect pad should succeed");
        assert_eq!(padded.width(), 5);
        assert_eq!(padded.height(), 5);

        // Interior should be preserved exactly
        for row in 0..3 {
            for col in 0..3 {
                let orig = img.get_pixel(row, col, 0).expect("test: orig pixel");
                let padded_val = padded
                    .get_pixel(row + 1, col + 1, 0)
                    .expect("test: padded pixel");
                assert!(
                    (orig - padded_val).abs() < 1e-10,
                    "Reflect pad interior at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    padded_val
                );
            }
        }
    }

    // ========================================================================
    // affine_transform tests
    // ========================================================================

    #[test]
    fn test_affine_identity() {
        let img = make_test_image(8, 8);
        let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&img, &matrix).expect("test: identity affine should succeed");
        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);

        // Interior pixels should be preserved
        for row in 1..7 {
            for col in 1..7 {
                let orig = img.get_pixel(row, col, 0).expect("test: orig pixel");
                let aff = result.get_pixel(row, col, 0).expect("test: affine pixel");
                assert!(
                    (orig - aff).abs() < 1e-6,
                    "Identity affine at ({}, {}): expected {}, got {}",
                    row,
                    col,
                    orig,
                    aff
                );
            }
        }
    }

    #[test]
    fn test_affine_translation() {
        let mut data = vec![0.0; 8 * 8];
        data[3 * 8 + 3] = 1.0; // bright pixel at (row=3, col=3)
        let img = Image::from_grayscale(8, 8, &data).expect("test: image creation should succeed");

        // Translation: src_x = col + 1, src_y = row + 1
        // This shifts the image up-left by 1 pixel
        let matrix = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let result =
            affine_transform(&img, &matrix).expect("test: translation affine should succeed");

        // The bright pixel at source (3,3) should appear at output (2,2)
        let shifted = result.get_pixel(2, 2, 0).expect("test: shifted pixel");
        assert!(
            (shifted - 1.0).abs() < 1e-6,
            "Translated bright pixel should appear at (2,2): got {}",
            shifted
        );
    }

    #[test]
    fn test_affine_scale() {
        let img = make_test_image(8, 8);
        // Scale by 0.5 (each output pixel maps to half the input coordinate)
        let matrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        let result = affine_transform(&img, &matrix).expect("test: scale affine should succeed");

        // Origin should be preserved
        let orig = img.get_pixel(0, 0, 0).expect("test: orig pixel");
        let scaled = result.get_pixel(0, 0, 0).expect("test: scaled pixel");
        assert!(
            (orig - scaled).abs() < 1e-6,
            "Affine scale origin: expected {}, got {}",
            orig,
            scaled
        );
    }

    // ========================================================================
    // Multi-channel tests
    // ========================================================================

    #[test]
    fn test_resize_rgb_image() {
        let data = vec![0.5; 4 * 4 * 3];
        let img = Image::from_rgb(4, 4, &data).expect("test: RGB image creation should succeed");
        let resized =
            resize_bilinear(&img, 8, 8).expect("test: RGB bilinear resize should succeed");
        assert_eq!(resized.width(), 8);
        assert_eq!(resized.height(), 8);
        assert_eq!(resized.channels(), 3);

        for row in 0..8 {
            for col in 0..8 {
                for ch in 0..3 {
                    let v = resized
                        .get_pixel(row, col, ch)
                        .expect("test: pixel access should succeed");
                    assert!(
                        (v - 0.5).abs() < 1e-6,
                        "RGB resize constant at ({}, {}, {}): got {}",
                        row,
                        col,
                        ch,
                        v
                    );
                }
            }
        }
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_reflect_coordinate_values() {
        // Interior values pass through
        assert_eq!(reflect_coordinate(3, 10), 3);
        assert_eq!(reflect_coordinate(0, 10), 0);
        assert_eq!(reflect_coordinate(9, 10), 9);
        // Negative indices reflect
        assert_eq!(reflect_coordinate(-1, 10), 1);
        assert_eq!(reflect_coordinate(-2, 10), 2);
        // Beyond-end indices reflect
        assert_eq!(reflect_coordinate(10, 10), 8);
        assert_eq!(reflect_coordinate(11, 10), 7);
        // Edge case: single-element dimension
        assert_eq!(reflect_coordinate(-1, 1), 0);
        assert_eq!(reflect_coordinate(1, 1), 0);
    }

    #[test]
    fn test_map_pad_coordinate_modes() {
        // Within bounds: all modes return the same result
        assert_eq!(map_pad_coordinate(5, 2, 10, PadMode::Constant), Some(3));
        assert_eq!(map_pad_coordinate(5, 2, 10, PadMode::Reflect), Some(3));
        assert_eq!(map_pad_coordinate(5, 2, 10, PadMode::Replicate), Some(3));
        assert_eq!(map_pad_coordinate(5, 2, 10, PadMode::Wrap), Some(3));

        // Before image: Constant returns None, Replicate returns 0
        assert_eq!(map_pad_coordinate(0, 2, 10, PadMode::Constant), None);
        assert_eq!(map_pad_coordinate(0, 2, 10, PadMode::Replicate), Some(0));

        // After image: Constant returns None, Replicate returns last index
        assert_eq!(map_pad_coordinate(15, 2, 10, PadMode::Constant), None);
        assert_eq!(map_pad_coordinate(15, 2, 10, PadMode::Replicate), Some(9));
    }
}

//! Morphological Operations
//!
//! Provides morphological image processing operations commonly used for
//! binary and grayscale image analysis. Morphological operations work by
//! applying a structuring element to an image to probe its shape.
//!
//! # Operations
//!
//! - **Erosion**: Shrinks bright regions, removes small bright spots
//! - **Dilation**: Expands bright regions, fills small dark holes
//! - **Opening**: Erosion followed by dilation (removes small bright features)
//! - **Closing**: Dilation followed by erosion (fills small dark features)
//! - **Morphological gradient**: Dilation minus erosion (outlines regions)
//! - **Top hat**: Original minus opening (extracts small bright features)
//! - **Black hat**: Closing minus original (extracts small dark features)
//!
//! # Structuring Elements
//!
//! Three standard shapes are provided:
//! - Rectangular (all ones)
//! - Cross (plus sign pattern)
//! - Elliptical (approximation of a circle/ellipse)
//!
//! # SCIRS2 Policy
//!
//! All implementations use `crate::array::Array` for data and follow
//! the pure Rust requirement.

use super::{ColorSpace, CvError, Image};
use crate::error::NumRs2Error;

/// Shape of a structuring element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuringElementShape {
    /// Rectangular structuring element (all positions active)
    Rectangular,
    /// Cross-shaped structuring element (only center row and column)
    Cross,
    /// Elliptical structuring element (approximation of a circle)
    Elliptical,
}

/// A structuring element for morphological operations.
///
/// The structuring element defines the neighborhood pattern used
/// for erosion and dilation operations. Active positions (true)
/// participate in the operation.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    /// The element data: true means active, false means inactive
    data: Vec<Vec<bool>>,
    /// Height of the structuring element
    height: usize,
    /// Width of the structuring element
    width: usize,
    /// Anchor row (typically center)
    anchor_row: usize,
    /// Anchor column (typically center)
    anchor_col: usize,
}

impl StructuringElement {
    /// Creates a new structuring element with the given shape and size.
    ///
    /// # Arguments
    /// * `shape` - The shape of the structuring element
    /// * `height` - Height (must be odd)
    /// * `width` - Width (must be odd)
    ///
    /// # Errors
    /// Returns error if dimensions are even or zero
    pub fn new(
        shape: StructuringElementShape,
        height: usize,
        width: usize,
    ) -> Result<Self, NumRs2Error> {
        if height == 0 || width == 0 {
            return Err(CvError::InvalidKernelSize(0).into());
        }
        if height.is_multiple_of(2) || width.is_multiple_of(2) {
            return Err(CvError::InvalidParameter(
                "Structuring element dimensions must be odd".to_string(),
            )
            .into());
        }

        let anchor_row = height / 2;
        let anchor_col = width / 2;

        let data = match shape {
            StructuringElementShape::Rectangular => {
                vec![vec![true; width]; height]
            }
            StructuringElementShape::Cross => {
                let mut d = vec![vec![false; width]; height];
                for i in 0..height {
                    d[i][anchor_col] = true;
                }
                for j in 0..width {
                    d[anchor_row][j] = true;
                }
                d
            }
            StructuringElementShape::Elliptical => {
                let mut d = vec![vec![false; width]; height];
                let a = anchor_col as f64; // semi-axis width
                let b = anchor_row as f64; // semi-axis height
                for i in 0..height {
                    for j in 0..width {
                        let dy = (i as f64 - b) / b.max(1.0);
                        let dx = (j as f64 - a) / a.max(1.0);
                        if dx * dx + dy * dy <= 1.0 + 1e-10 {
                            d[i][j] = true;
                        }
                    }
                }
                d
            }
        };

        Ok(Self {
            data,
            height,
            width,
            anchor_row,
            anchor_col,
        })
    }

    /// Creates a square structuring element of the given size.
    ///
    /// # Arguments
    /// * `shape` - The shape of the structuring element
    /// * `size` - Size of the structuring element (must be odd)
    ///
    /// # Errors
    /// Returns error if size is even or zero
    pub fn square(shape: StructuringElementShape, size: usize) -> Result<Self, NumRs2Error> {
        Self::new(shape, size, size)
    }

    /// Returns whether the given position in the structuring element is active.
    pub fn is_active(&self, row: usize, col: usize) -> bool {
        if row < self.height && col < self.width {
            self.data[row][col]
        } else {
            false
        }
    }

    /// Returns the height of the structuring element.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the width of the structuring element.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the anchor row.
    pub fn anchor_row(&self) -> usize {
        self.anchor_row
    }

    /// Returns the anchor column.
    pub fn anchor_col(&self) -> usize {
        self.anchor_col
    }

    /// Returns the number of active (true) positions.
    pub fn active_count(&self) -> usize {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v)
            .count()
    }
}

/// Fetches a pixel value with boundary handling for morphological operations.
fn fetch_morph_pixel(img: &Image, row: isize, col: isize, default_val: f64) -> f64 {
    let h = img.height() as isize;
    let w = img.width() as isize;
    if row < 0 || row >= h || col < 0 || col >= w {
        default_val
    } else {
        img.get_pixel(row as usize, col as usize, 0)
            .unwrap_or(default_val)
    }
}

/// Applies erosion to a grayscale image.
///
/// Erosion replaces each pixel with the minimum value among its neighbors
/// defined by the structuring element. This shrinks bright regions and
/// removes small bright features.
///
/// For binary images: a pixel is kept only if all structuring element
/// positions match foreground pixels.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element defining the neighborhood
///
/// # Returns
/// A new eroded image
///
/// # Errors
/// Returns error if the image is not grayscale
pub fn erosion(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }

    let h = img.height();
    let w = img.width();
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let mut min_val = f64::INFINITY;
            for si in 0..se.height() {
                for sj in 0..se.width() {
                    if se.is_active(si, sj) {
                        let img_row = row as isize + si as isize - se.anchor_row() as isize;
                        let img_col = col as isize + sj as isize - se.anchor_col() as isize;
                        // For erosion, out-of-bounds pixels default to infinity (max possible)
                        // so they don't affect the minimum
                        let pixel = fetch_morph_pixel(img, img_row, img_col, f64::INFINITY);
                        if pixel < min_val {
                            min_val = pixel;
                        }
                    }
                }
            }
            // If no active elements, keep original value
            if min_val.is_infinite() {
                min_val = img.get_pixel(row, col, 0).map_err(|e| {
                    NumRs2Error::ComputationError(format!("Erosion pixel read: {}", e))
                })?;
            }
            result
                .set_pixel(row, col, 0, min_val)
                .map_err(|e| NumRs2Error::ComputationError(format!("Erosion pixel set: {}", e)))?;
        }
    }

    Ok(result)
}

/// Applies dilation to a grayscale image.
///
/// Dilation replaces each pixel with the maximum value among its neighbors
/// defined by the structuring element. This expands bright regions and
/// fills small dark holes.
///
/// For binary images: a pixel becomes foreground if any structuring element
/// position overlaps a foreground pixel.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element defining the neighborhood
///
/// # Returns
/// A new dilated image
///
/// # Errors
/// Returns error if the image is not grayscale
pub fn dilation(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }

    let h = img.height();
    let w = img.width();
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let mut max_val = f64::NEG_INFINITY;
            for si in 0..se.height() {
                for sj in 0..se.width() {
                    if se.is_active(si, sj) {
                        let img_row = row as isize + si as isize - se.anchor_row() as isize;
                        let img_col = col as isize + sj as isize - se.anchor_col() as isize;
                        // For dilation, out-of-bounds pixels default to -infinity (min possible)
                        // so they don't affect the maximum
                        let pixel = fetch_morph_pixel(img, img_row, img_col, f64::NEG_INFINITY);
                        if pixel > max_val {
                            max_val = pixel;
                        }
                    }
                }
            }
            // If no active elements, keep original value
            if max_val.is_infinite() && max_val < 0.0 {
                max_val = img.get_pixel(row, col, 0).map_err(|e| {
                    NumRs2Error::ComputationError(format!("Dilation pixel read: {}", e))
                })?;
            }
            result
                .set_pixel(row, col, 0, max_val)
                .map_err(|e| NumRs2Error::ComputationError(format!("Dilation pixel set: {}", e)))?;
        }
    }

    Ok(result)
}

/// Applies morphological opening: erosion followed by dilation.
///
/// Opening removes small bright features while preserving the overall shape.
/// It is useful for removing noise from binary images.
///
/// `opening(img, se) = dilation(erosion(img, se), se)`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element
///
/// # Returns
/// A new opened image
pub fn opening(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    let eroded = erosion(img, se)?;
    dilation(&eroded, se)
}

/// Applies morphological closing: dilation followed by erosion.
///
/// Closing fills small dark holes while preserving the overall shape.
/// It is useful for filling gaps in binary images.
///
/// `closing(img, se) = erosion(dilation(img, se), se)`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element
///
/// # Returns
/// A new closed image
pub fn closing(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    let dilated = dilation(img, se)?;
    erosion(&dilated, se)
}

/// Computes the morphological gradient: dilation minus erosion.
///
/// The morphological gradient highlights edges by computing the
/// difference between dilation and erosion. This produces an outline
/// of objects in the image.
///
/// `gradient(img, se) = dilation(img, se) - erosion(img, se)`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element
///
/// # Returns
/// A new image containing the morphological gradient
pub fn morphological_gradient(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    let dilated = dilation(img, se)?;
    let eroded = erosion(img, se)?;

    let h = img.height();
    let w = img.width();
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let d = dilated.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Gradient dilation read: {}", e))
            })?;
            let e = eroded.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Gradient erosion read: {}", e))
            })?;
            result
                .set_pixel(row, col, 0, d - e)
                .map_err(|err| NumRs2Error::ComputationError(format!("Gradient set: {}", err)))?;
        }
    }

    Ok(result)
}

/// Computes the top hat transform: original minus opening.
///
/// The top hat extracts small bright features on a dark background.
/// It enhances features that are smaller than the structuring element.
///
/// `top_hat(img, se) = img - opening(img, se)`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element
///
/// # Returns
/// A new image containing the top hat result
pub fn top_hat(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    let opened = opening(img, se)?;

    let h = img.height();
    let w = img.width();
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let orig = img.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Top hat original read: {}", e))
            })?;
            let op = opened.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Top hat opened read: {}", e))
            })?;
            result
                .set_pixel(row, col, 0, (orig - op).max(0.0))
                .map_err(|e| NumRs2Error::ComputationError(format!("Top hat set: {}", e)))?;
        }
    }

    Ok(result)
}

/// Computes the black hat transform: closing minus original.
///
/// The black hat extracts small dark features on a bright background.
/// It enhances dark features that are smaller than the structuring element.
///
/// `black_hat(img, se) = closing(img, se) - img`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `se` - Structuring element
///
/// # Returns
/// A new image containing the black hat result
pub fn black_hat(img: &Image, se: &StructuringElement) -> Result<Image, NumRs2Error> {
    let closed = closing(img, se)?;

    let h = img.height();
    let w = img.width();
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let cl = closed.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Black hat closed read: {}", e))
            })?;
            let orig = img.get_pixel(row, col, 0).map_err(|e| {
                NumRs2Error::ComputationError(format!("Black hat original read: {}", e))
            })?;
            result
                .set_pixel(row, col, 0, (cl - orig).max(0.0))
                .map_err(|e| NumRs2Error::ComputationError(format!("Black hat set: {}", e)))?;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a binary-like test image with a bright square in the center.
    fn make_binary_image(size: usize) -> Image {
        let mut data = vec![0.0; size * size];
        let margin = size / 4;
        for row in margin..(size - margin) {
            for col in margin..(size - margin) {
                data[row * size + col] = 1.0;
            }
        }
        Image::from_grayscale(size, size, &data).expect("test: image creation should succeed")
    }

    #[test]
    fn test_structuring_element_rectangular() {
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        assert_eq!(se.height(), 3);
        assert_eq!(se.width(), 3);
        assert_eq!(se.active_count(), 9);
        assert!(se.is_active(0, 0));
        assert!(se.is_active(1, 1));
        assert!(se.is_active(2, 2));
    }

    #[test]
    fn test_structuring_element_cross() {
        let se = StructuringElement::square(StructuringElementShape::Cross, 3)
            .expect("test: SE creation should succeed");
        assert_eq!(se.active_count(), 5);
        // Center row and column should be active
        assert!(se.is_active(1, 0));
        assert!(se.is_active(1, 1));
        assert!(se.is_active(1, 2));
        assert!(se.is_active(0, 1));
        assert!(se.is_active(2, 1));
        // Corners should be inactive
        assert!(!se.is_active(0, 0));
        assert!(!se.is_active(0, 2));
        assert!(!se.is_active(2, 0));
        assert!(!se.is_active(2, 2));
    }

    #[test]
    fn test_structuring_element_elliptical() {
        let se = StructuringElement::square(StructuringElementShape::Elliptical, 5)
            .expect("test: SE creation should succeed");
        assert!(se.is_active(2, 2)); // center always active
                                     // Active count should be less than full rectangular
        assert!(se.active_count() < 25);
        assert!(se.active_count() > 5);
    }

    #[test]
    fn test_structuring_element_invalid_size() {
        let result = StructuringElement::square(StructuringElementShape::Rectangular, 4);
        assert!(result.is_err());
        let result = StructuringElement::square(StructuringElementShape::Rectangular, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_erosion_shrinks_bright_region() {
        let img = make_binary_image(16);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let eroded = erosion(&img, &se).expect("test: erosion should succeed");

        // The bright region should be smaller after erosion
        let original_bright: usize = img.to_vec().iter().filter(|&&v| v > 0.5).count();
        let eroded_bright: usize = eroded.to_vec().iter().filter(|&&v| v > 0.5).count();
        assert!(
            eroded_bright < original_bright,
            "Erosion should shrink the bright region: {} vs {}",
            eroded_bright,
            original_bright
        );
    }

    #[test]
    fn test_dilation_expands_bright_region() {
        let img = make_binary_image(16);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let dilated = dilation(&img, &se).expect("test: dilation should succeed");

        let original_bright: usize = img.to_vec().iter().filter(|&&v| v > 0.5).count();
        let dilated_bright: usize = dilated.to_vec().iter().filter(|&&v| v > 0.5).count();
        assert!(
            dilated_bright > original_bright,
            "Dilation should expand the bright region: {} vs {}",
            dilated_bright,
            original_bright
        );
    }

    #[test]
    fn test_opening_idempotent() {
        // Opening is idempotent: opening(opening(img)) == opening(img)
        let img = make_binary_image(16);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let opened_once = opening(&img, &se).expect("test: opening should succeed");
        let opened_twice = opening(&opened_once, &se).expect("test: opening should succeed");

        let data_once = opened_once.to_vec();
        let data_twice = opened_twice.to_vec();
        for (v1, v2) in data_once.iter().zip(data_twice.iter()) {
            assert!((v1 - v2).abs() < 1e-10, "Opening should be idempotent");
        }
    }

    #[test]
    fn test_closing_idempotent() {
        // Closing is idempotent: closing(closing(img)) == closing(img)
        let img = make_binary_image(16);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let closed_once = closing(&img, &se).expect("test: closing should succeed");
        let closed_twice = closing(&closed_once, &se).expect("test: closing should succeed");

        let data_once = closed_once.to_vec();
        let data_twice = closed_twice.to_vec();
        for (v1, v2) in data_once.iter().zip(data_twice.iter()) {
            assert!((v1 - v2).abs() < 1e-10, "Closing should be idempotent");
        }
    }

    #[test]
    fn test_morphological_gradient() {
        let img = make_binary_image(16);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let gradient =
            morphological_gradient(&img, &se).expect("test: morphological gradient should succeed");

        // Interior of bright region should have gradient ~0
        let center = gradient
            .get_pixel(8, 8, 0)
            .expect("test: pixel read should succeed");
        assert!(
            center.abs() < 1e-10,
            "Gradient should be zero in constant interior: got {}",
            center
        );

        // At edges, gradient should be positive
        let gradient_data = gradient.to_vec();
        let max_gradient = gradient_data.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_gradient > 0.0,
            "Gradient should have positive values at edges"
        );
    }

    #[test]
    fn test_top_hat_on_constant() {
        // Top hat of a constant image should be all zeros
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let th = top_hat(&img, &se).expect("test: top hat should succeed");

        for row in 1..15 {
            for col in 1..15 {
                let v = th
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    v.abs() < 1e-8,
                    "Top hat of constant should be zero at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_black_hat_on_constant() {
        // Black hat of a constant image should be all zeros
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let bh = black_hat(&img, &se).expect("test: black hat should succeed");

        for row in 1..15 {
            for col in 1..15 {
                let v = bh
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    v.abs() < 1e-8,
                    "Black hat of constant should be zero at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_erosion_then_dilation_preserves_shape() {
        // For a large enough bright region, opening should approximately
        // preserve the shape (erosion then dilation)
        let img = make_binary_image(32);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let opened = opening(&img, &se).expect("test: opening should succeed");

        // Center pixel should still be bright
        let center = opened
            .get_pixel(16, 16, 0)
            .expect("test: pixel read should succeed");
        assert!(
            center > 0.9,
            "Opening should preserve center of large bright region: got {}",
            center
        );
    }

    #[test]
    fn test_erosion_on_all_zeros() {
        let img = Image::zeros_grayscale(8, 8);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let eroded = erosion(&img, &se).expect("test: erosion should succeed");

        for v in eroded.to_vec() {
            assert!(
                v.abs() < 1e-10,
                "Erosion of all-zero image should remain zero"
            );
        }
    }

    #[test]
    fn test_dilation_on_all_zeros() {
        let img = Image::zeros_grayscale(8, 8);
        let se = StructuringElement::square(StructuringElementShape::Rectangular, 3)
            .expect("test: SE creation should succeed");
        let dilated = dilation(&img, &se).expect("test: dilation should succeed");

        for v in dilated.to_vec() {
            assert!(
                v.abs() < 1e-10,
                "Dilation of all-zero image should remain zero"
            );
        }
    }
}

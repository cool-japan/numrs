//! N-Dimensional Image Processing Module
//!
//! This module provides comprehensive multidimensional image processing functionality,
//! built on top of `scirs2-ndimage`. It includes:
//!
//! - **Filters**: Gaussian, median, rank, and edge detection filters
//! - **Morphology**: Binary and grayscale morphological operations
//! - **Measurements**: Region properties, moments, and extrema detection
//! - **Segmentation**: Thresholding and watershed algorithms
//! - **Features**: Corner and edge detection
//! - **Interpolation**: Spline and geometric transformations
//!
//! # Examples
//!
//! ```
//! use numrs2::ndimage;
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a sample 2D image
//! let image = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| {
//!     if (i > 3 && i < 7) && (j > 3 && j < 7) {
//!         1.0
//!     } else {
//!         0.0
//!     }
//! });
//!
//! // Apply Gaussian filter
//! let sigma = 1.0;
//! let filtered = ndimage::filters::gaussian_filter(&image, sigma, None, None).expect("gaussian_filter should succeed");
//!
//! // Create binary image and apply binary dilation
//! let binary_image: Array2<bool> = Array2::from_shape_fn((10, 10), |(i, j)| {
//!     i > 3 && i < 7 && j > 3 && j < 7
//! });
//! let dilated = ndimage::morphology::binary_dilation(&binary_image, None, None, None, None, None, None).expect("binary_dilation should succeed");
//! ```
//!
//! # Filters
//!
//! The filters module provides various image filtering operations:
//!
//! - `gaussian_filter`: Apply Gaussian filter for smoothing
//! - `median_filter`: Apply median filter for noise reduction
//! - `minimum_filter`, `maximum_filter`: Morphological filters
//! - `sobel`, `prewitt`: Edge detection filters
//! - `laplace`: Laplace filter for edge enhancement
//!
//! # Morphology
//!
//! Morphological operations for binary and grayscale images:
//!
//! - `binary_erosion`, `binary_dilation`: Basic morphological operations
//! - `binary_opening`, `binary_closing`: Composite operations
//! - `binary_fill_holes`: Fill holes in binary objects
//! - `grey_erosion`, `grey_dilation`: Grayscale morphology
//!
//! # Measurements
//!
//! Functions for measuring properties of labeled regions:
//!
//! - `label`: Connected component labeling
//! - `center_of_mass`: Compute center of mass
//! - `sum`, `mean`, `variance`: Statistics over labeled regions
//! - `minimum`, `maximum`: Extrema over labeled regions
//!
//! # Segmentation
//!
//! Image segmentation algorithms:
//!
//! - `threshold_otsu`: Otsu's automatic thresholding
//! - `threshold_adaptive`: Adaptive local thresholding
//! - `watershed`: Watershed segmentation
//!
//! # Features
//!
//! Feature detection algorithms:
//!
//! - `corner_harris`: Harris corner detector
//! - `canny`: Canny edge detector
//! - `sobel`: Sobel edge detector
//!
//! # Interpolation
//!
//! Geometric transformations and interpolation:
//!
//! - `rotate`, `shift`, `zoom`: Basic geometric transformations
//! - `affine_transform`: General affine transformations
//! - `map_coordinates`: Custom coordinate mapping

// Re-export all scirs2-ndimage modules
pub use scirs2_ndimage::*;

// Additional NumRS2-specific convenience functions and aliases can be added here

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_gaussian_filter_basic() {
        // Create a simple 5x5 image with a bright center
        let image =
            Array2::<f64>::from_shape_fn((5, 5), |(i, j)| if i == 2 && j == 2 { 1.0 } else { 0.0 });

        // Apply Gaussian filter
        let sigma = 1.0;
        let result = filters::gaussian_filter(&image, sigma, None, None)
            .expect("gaussian_filter should succeed");

        // Center should still be brightest
        assert!(result[[2, 2]] > result[[0, 0]]);
        assert!(result[[2, 2]] > result[[4, 4]]);
    }

    #[test]
    fn test_binary_dilation_basic() {
        // Create a simple binary image
        let mut image = Array2::<bool>::from_elem((5, 5), false);
        image[[2, 2]] = true;

        // Create structuring element (3x3 square)
        let struct_elem = Array2::<bool>::from_elem((3, 3), true);

        // Apply dilation (7 parameters: input, structure, iterations, mask, border_value, origin, brute_force)
        let result =
            morphology::binary_dilation(&image, Some(&struct_elem), None, None, None, None, None)
                .expect("binary_dilation should succeed");

        // Check that the center pixel and its neighbors are true
        assert!(result[[2, 2]]);
        assert!(result[[1, 2]]);
        assert!(result[[3, 2]]);
        assert!(result[[2, 1]]);
        assert!(result[[2, 3]]);
    }

    #[test]
    fn test_label_basic() {
        // Create an image with two separate objects
        let mut image = Array2::<bool>::from_elem((5, 5), false);
        image[[1, 1]] = true;
        image[[1, 2]] = true;
        image[[3, 3]] = true;
        image[[3, 4]] = true;

        // Label connected components (4 parameters: input, structure, connectivity, background)
        let (labeled, num_features) =
            morphology::label(&image, None, None, None).expect("label should succeed");

        // Should find at least some labeled components
        assert!(num_features > 0);

        // Check that pixels with true values got labeled
        assert_ne!(labeled[[1, 1]], 0);
        assert_ne!(labeled[[3, 3]], 0);
    }

    #[test]
    fn test_center_of_mass() {
        // Create a simple image with known center of mass
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Center of mass should be at (2, 2)
        let com = measurements::center_of_mass(&image).expect("center_of_mass should succeed");

        assert!((com[0] - 2.0).abs() < 1e-10);
        assert!((com[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_filter_basic() {
        // Create an image with salt-and-pepper noise
        let mut image = Array2::<f64>::from_elem((5, 5), 0.5);
        image[[0, 0]] = 0.0; // Salt
        image[[4, 4]] = 1.0; // Pepper

        // Apply median filter with size as slice
        let result =
            filters::median_filter(&image, &[3, 3], None).expect("median_filter should succeed");

        // The corners should be smoothed out
        assert!((result[[0, 0]] - 0.5).abs() < 0.2);
        assert!((result[[4, 4]] - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_sobel_edge_detection() {
        // Create an image with a vertical edge
        let image = Array2::<f64>::from_shape_fn((5, 5), |(_, j)| if j < 2 { 0.0 } else { 1.0 });

        // Apply Sobel filter (axis 1 for detecting vertical edges)
        let result = filters::sobel(&image, 1, None).expect("sobel should succeed");

        // Middle column should have high gradient values
        assert!(result[[2, 2]].abs() > 0.1);
    }

    #[test]
    fn test_binary_erosion() {
        // Create a binary image with a small object
        let mut image = Array2::<bool>::from_elem((5, 5), false);
        for i in 1..4 {
            for j in 1..4 {
                image[[i, j]] = true;
            }
        }

        // Create structuring element (3x3 square)
        let struct_elem = Array2::<bool>::from_elem((3, 3), true);

        // Apply erosion (7 parameters: input, structure, iterations, mask, border_value, origin, brute_force)
        let result =
            morphology::binary_erosion(&image, Some(&struct_elem), None, None, None, None, None)
                .expect("binary_erosion should succeed");

        // Only the center pixel should remain
        assert!(result[[2, 2]]);
        assert!(!result[[1, 1]]);
        assert!(!result[[3, 3]]);
    }

    #[test]
    fn test_disk_structuring_element() {
        // Generate a disk-shaped structuring element
        let disk = morphology::disk_structure(2.0, Some(2)).expect("disk_structure should succeed");

        // Check that it exists and has reasonable size
        assert!(disk.ndim() == 2);
        assert!(!disk.is_empty());
    }
}

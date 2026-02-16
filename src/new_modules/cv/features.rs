//! Feature Detection and Description
//!
//! Provides feature detection algorithms for identifying interesting points
//! (corners, blobs) in images, along with descriptor extraction for matching.
//!
//! # Algorithms
//!
//! - **Harris corner detector**: Classic corner detection using the structure tensor
//! - **FAST corner detector**: Features from Accelerated Segment Test
//! - **Non-maximum suppression**: Post-processing to select local maxima
//! - **Simple descriptor**: Patch-based descriptor extraction with gradient histograms
//! - **Brute force matching**: Descriptor matching using L2 distance
//!
//! # SCIRS2 Policy
//!
//! All implementations use `crate::array::Array` for data and follow
//! the pure Rust requirement.

use super::filters::{gaussian_blur, sobel_x, sobel_y};
use super::{ColorSpace, CvError, Image};
use crate::error::NumRs2Error;

/// A detected keypoint in an image.
///
/// Represents a point of interest found by a feature detector, along with
/// metadata about the detection (response strength, scale, orientation).
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// Row position (y coordinate)
    pub row: usize,
    /// Column position (x coordinate)
    pub col: usize,
    /// Response strength (higher = stronger feature)
    pub response: f64,
    /// Scale at which the feature was detected (default 1.0)
    pub scale: f64,
    /// Orientation angle in radians (0.0 if not computed)
    pub orientation: f64,
}

impl Keypoint {
    /// Creates a new keypoint at the given position with the given response.
    pub fn new(row: usize, col: usize, response: f64) -> Self {
        Self {
            row,
            col,
            response,
            scale: 1.0,
            orientation: 0.0,
        }
    }

    /// Creates a new keypoint with all parameters specified.
    pub fn with_full(row: usize, col: usize, response: f64, scale: f64, orientation: f64) -> Self {
        Self {
            row,
            col,
            response,
            scale,
            orientation,
        }
    }
}

/// A feature descriptor associated with a keypoint.
///
/// The descriptor is a fixed-length vector of floating-point values
/// that characterizes the local image neighborhood around a keypoint.
/// Descriptors are normalized to unit length for illumination invariance.
#[derive(Debug, Clone)]
pub struct FeatureDescriptor {
    /// The keypoint this descriptor is associated with
    pub keypoint: Keypoint,
    /// The descriptor vector (normalized to unit length)
    pub descriptor: Vec<f64>,
}

/// A match between two feature descriptors.
///
/// Represents a correspondence between a query descriptor and a train
/// (reference) descriptor, along with the L2 distance between them.
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    /// Index of the query descriptor
    pub query_idx: usize,
    /// Index of the train (reference) descriptor
    pub train_idx: usize,
    /// Distance between the matched descriptors (L2)
    pub distance: f64,
}

/// Applies the Harris corner detector to a grayscale image.
///
/// The Harris corner detector uses the structure tensor (second moment matrix)
/// to identify corners. At each pixel, the matrix:
///
/// ```text
/// M = [ sum(Ix^2)   sum(Ix*Iy) ]
///     [ sum(Ix*Iy)  sum(Iy^2)  ]
/// ```
///
/// is computed over a window of size `block_size`, where Ix and Iy are image
/// gradients obtained via the Sobel operator.
///
/// The corner response function is:
///
/// `R = det(M) - k_sensitivity * trace(M)^2`
///
/// where `k_sensitivity` is typically in the range `[0.04, 0.06]`.
///
/// Pixels with response above `threshold` are returned as keypoints,
/// sorted by descending response strength.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `block_size` - Size of the neighborhood for computing the structure tensor
/// * `k_sensitivity` - Harris detector sensitivity parameter (typical: 0.04-0.06)
/// * `threshold` - Minimum response to consider a pixel as a corner
///
/// # Returns
/// A vector of detected keypoints sorted by response strength (descending)
///
/// # Errors
/// Returns error if the image is not grayscale or block_size is zero
pub fn harris_corner_detect(
    img: &Image,
    block_size: usize,
    k_sensitivity: f64,
    threshold: f64,
) -> Result<Vec<Keypoint>, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if block_size == 0 {
        return Err(
            CvError::InvalidParameter("block_size must be greater than zero".to_string()).into(),
        );
    }

    let h = img.height();
    let w = img.width();
    let half = (block_size / 2) as isize;

    // Apply a small Gaussian pre-smoothing for noise robustness
    let smoothed = if h >= 5 && w >= 5 {
        gaussian_blur(img, 3, 0.8)?
    } else {
        img.clone()
    };

    // Compute gradients using Sobel operators
    let gx = sobel_x(&smoothed)?;
    let gy = sobel_y(&smoothed)?;

    // Compute structure tensor components and Harris response
    let mut keypoints = Vec::new();

    for row in 0..h {
        for col in 0..w {
            let mut sum_ix_sq = 0.0;
            let mut sum_iy_sq = 0.0;
            let mut sum_ix_iy = 0.0;

            // Accumulate over the block_size neighborhood
            for di in -half..=half {
                for dj in -half..=half {
                    let nr = row as isize + di;
                    let nc = col as isize + dj;
                    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                        let ix = gx.get_pixel(nr as usize, nc as usize, 0).unwrap_or(0.0);
                        let iy = gy.get_pixel(nr as usize, nc as usize, 0).unwrap_or(0.0);
                        sum_ix_sq += ix * ix;
                        sum_iy_sq += iy * iy;
                        sum_ix_iy += ix * iy;
                    }
                }
            }

            // Harris response: R = det(M) - k * trace(M)^2
            let det = sum_ix_sq * sum_iy_sq - sum_ix_iy * sum_ix_iy;
            let trace = sum_ix_sq + sum_iy_sq;
            let response = det - k_sensitivity * trace * trace;

            if response > threshold {
                keypoints.push(Keypoint::new(row, col, response));
            }
        }
    }

    // Sort by response (descending)
    keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(keypoints)
}

/// Applies the FAST (Features from Accelerated Segment Test) corner detector.
///
/// FAST tests whether a pixel is a corner by examining 16 pixels on a
/// Bresenham circle of radius 3 around the candidate. A corner is detected
/// if at least `n_consecutive` contiguous pixels on the circle are all
/// brighter than the center pixel by `threshold`, or all darker by `threshold`.
///
/// The algorithm uses a high-speed test on 4 cardinal points for early
/// rejection before performing the full contiguity check.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `threshold` - Intensity difference threshold (e.g., 0.05 for normalized images)
/// * `n_consecutive` - Number of contiguous pixels required (typically 9-12)
///
/// # Returns
/// A vector of detected keypoints sorted by response (descending)
///
/// # Errors
/// Returns error if image is not grayscale or parameters are invalid
pub fn fast_corner_detect(
    img: &Image,
    threshold: f64,
    n_consecutive: usize,
) -> Result<Vec<Keypoint>, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if !(1..=16).contains(&n_consecutive) {
        return Err(CvError::InvalidParameter(
            "n_consecutive must be between 1 and 16".to_string(),
        )
        .into());
    }
    if threshold < 0.0 {
        return Err(CvError::InvalidParameter("threshold must be non-negative".to_string()).into());
    }

    let h = img.height();
    let w = img.width();

    // 16-pixel Bresenham circle of radius 3
    // Positions relative to center pixel (row_offset, col_offset)
    let circle: [(isize, isize); 16] = [
        (-3, 0),  // 0: top
        (-3, 1),  // 1
        (-2, 2),  // 2
        (-1, 3),  // 3
        (0, 3),   // 4: right
        (1, 3),   // 5
        (2, 2),   // 6
        (3, 1),   // 7
        (3, 0),   // 8: bottom
        (3, -1),  // 9
        (2, -2),  // 10
        (1, -3),  // 11
        (0, -3),  // 12: left
        (-1, -3), // 13
        (-2, -2), // 14
        (-3, -1), // 15
    ];

    let mut keypoints = Vec::new();

    // Need margin of 3 pixels for the circle
    let margin = 3_usize;
    if h <= 2 * margin || w <= 2 * margin {
        return Ok(keypoints);
    }

    for row in margin..(h - margin) {
        for col in margin..(w - margin) {
            let center = img
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("FAST center: {}", e)))?;
            let bright_thresh = center + threshold;
            let dark_thresh = center - threshold;

            // High-speed test: check pixels at indices 0, 4, 8, 12 (N, E, S, W)
            // The minimum number of cardinal pixels that must match scales with
            // n_consecutive: an arc of n contiguous pixels on a 16-point circle
            // is guaranteed to contain at least floor(n/4) of the 4 cardinal points.
            // Standard FAST-9 traditionally uses 3, but the correct geometric
            // lower bound is floor(n/4), clamped to at least 1.
            let min_cardinal = (n_consecutive / 4).max(1) as u8;
            let test_indices = [0_usize, 4, 8, 12];
            let mut bright_count_quick = 0_u8;
            let mut dark_count_quick = 0_u8;
            for &idx in &test_indices {
                let (dr, dc) = circle[idx];
                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;
                let val = img.get_pixel(nr, nc, 0).unwrap_or(center);
                if val > bright_thresh {
                    bright_count_quick += 1;
                } else if val < dark_thresh {
                    dark_count_quick += 1;
                }
            }

            // Quick rejection: need at least min_cardinal of the 4 cardinal pixels
            if bright_count_quick < min_cardinal && dark_count_quick < min_cardinal {
                continue;
            }

            // Full test: check for n contiguous brighter or darker pixels
            // We duplicate the circle for wrap-around checking
            let mut brighter = [false; 32];
            let mut darker = [false; 32];

            for p in 0..16 {
                let (dr, dc) = circle[p];
                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;
                let val = img.get_pixel(nr, nc, 0).unwrap_or(center);
                if val > bright_thresh {
                    brighter[p] = true;
                    brighter[p + 16] = true;
                }
                if val < dark_thresh {
                    darker[p] = true;
                    darker[p + 16] = true;
                }
            }

            let is_bright_corner = has_n_contiguous(&brighter, n_consecutive);
            let is_dark_corner = has_n_contiguous(&darker, n_consecutive);

            if is_bright_corner || is_dark_corner {
                // Compute corner response: sum of absolute differences from
                // all 16 circle pixels, weighted by whether they are part
                // of the contiguous arc
                let mut response = 0.0;
                for p in 0..16 {
                    let (dr, dc) = circle[p];
                    let nr = (row as isize + dr) as usize;
                    let nc = (col as isize + dc) as usize;
                    let val = img.get_pixel(nr, nc, 0).unwrap_or(center);
                    response += (val - center).abs();
                }
                keypoints.push(Keypoint::new(row, col, response));
            }
        }
    }

    // Sort by response (descending)
    keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(keypoints)
}

/// Checks if there are at least `n` contiguous true values in a doubled circular array.
///
/// The array should be of length 32 (16 circle positions doubled for wrap-around).
fn has_n_contiguous(arr: &[bool], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let len = arr.len();
    // Only need to check up to len/2 + n to cover all wrap-around cases
    let check_len = len.min(len / 2 + n);
    let mut consecutive = 0_usize;
    for i in 0..check_len {
        if arr[i] {
            consecutive += 1;
            if consecutive >= n {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

/// Applies non-maximum suppression to a set of keypoints.
///
/// Removes keypoints that are not local maxima within the specified radius.
/// This reduces clusters of detections to a single strong detection. The
/// algorithm processes keypoints in order of decreasing response, keeping
/// each keypoint only if no stronger keypoint has already been kept within
/// the suppression radius.
///
/// # Arguments
/// * `keypoints` - Input keypoints (should be sorted by response, descending)
/// * `radius` - Suppression radius in pixels
///
/// # Returns
/// A filtered set of keypoints with non-maxima removed
pub fn non_maximum_suppression(keypoints: &[Keypoint], radius: f64) -> Vec<Keypoint> {
    if keypoints.is_empty() {
        return Vec::new();
    }

    let radius_sq = radius * radius;
    let mut suppressed = vec![false; keypoints.len()];
    let mut result = Vec::new();

    // Process in order of decreasing response (assumes sorted input)
    for i in 0..keypoints.len() {
        if suppressed[i] {
            continue;
        }
        result.push(keypoints[i].clone());

        // Suppress all weaker keypoints within the radius
        for j in (i + 1)..keypoints.len() {
            if suppressed[j] {
                continue;
            }
            let dr = keypoints[i].row as f64 - keypoints[j].row as f64;
            let dc = keypoints[i].col as f64 - keypoints[j].col as f64;
            let dist_sq = dr * dr + dc * dc;
            if dist_sq < radius_sq {
                suppressed[j] = true;
            }
        }
    }

    result
}

/// Extracts simple patch-based descriptors for a set of keypoints.
///
/// For each keypoint, a square patch of `patch_size x patch_size` around the
/// keypoint is analyzed using gradient orientation histograms. The patch is
/// divided into a 4x4 grid of cells, and for each cell an 8-bin histogram
/// of gradient orientations is computed. This yields a 128-dimensional
/// descriptor vector (4 * 4 * 8 = 128) similar to SIFT.
///
/// The descriptor is L2-normalized, clamped at 0.2, and re-normalized for
/// robustness against illumination changes.
///
/// Keypoints too close to the image border (where the full patch cannot be
/// extracted) are silently skipped.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `keypoints` - The keypoints to describe
/// * `patch_size` - Size of the descriptor patch (should be divisible by 4, default: 16)
///
/// # Returns
/// A vector of feature descriptors (may be shorter than input keypoints
/// if some are too close to the border)
///
/// # Errors
/// Returns error if the image is not grayscale
pub fn simple_descriptor(
    img: &Image,
    keypoints: &[Keypoint],
    patch_size: usize,
) -> Result<Vec<FeatureDescriptor>, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if patch_size == 0 {
        return Err(
            CvError::InvalidParameter("patch_size must be greater than zero".to_string()).into(),
        );
    }

    let h = img.height();
    let w = img.width();
    let half = patch_size / 2;
    let n_bins = 8_usize;
    let cells_per_side = 4_usize;
    let cell_size = patch_size / cells_per_side;

    if cell_size == 0 {
        return Err(CvError::InvalidParameter(
            "patch_size too small for 4x4 cell grid".to_string(),
        )
        .into());
    }

    // Pre-compute gradients once for the entire image
    let gx = sobel_x(img)?;
    let gy = sobel_y(img)?;

    let desc_len = cells_per_side * cells_per_side * n_bins;
    let mut descriptors = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        // Skip keypoints too close to the border
        if kp.row < half || kp.row + half >= h || kp.col < half || kp.col + half >= w {
            continue;
        }

        let mut descriptor = vec![0.0_f64; desc_len];

        // Compute gradient orientation histogram for each cell
        for ci in 0..cells_per_side {
            for cj in 0..cells_per_side {
                let cell_start_row = kp.row - half + ci * cell_size;
                let cell_start_col = kp.col - half + cj * cell_size;

                for di in 0..cell_size {
                    for dj in 0..cell_size {
                        let r = cell_start_row + di;
                        let c = cell_start_col + dj;

                        let ix = gx.get_pixel(r, c, 0).unwrap_or(0.0);
                        let iy = gy.get_pixel(r, c, 0).unwrap_or(0.0);

                        let magnitude = (ix * ix + iy * iy).sqrt();
                        let angle = iy.atan2(ix); // range [-pi, pi]

                        // Map angle to bin index [0, n_bins)
                        let normalized_angle =
                            (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
                        let bin = (normalized_angle * n_bins as f64).floor() as usize;
                        let bin = bin.min(n_bins - 1);

                        let cell_idx = ci * cells_per_side + cj;
                        let desc_idx = cell_idx * n_bins + bin;
                        if desc_idx < desc_len {
                            descriptor[desc_idx] += magnitude;
                        }
                    }
                }
            }
        }

        // L2-normalize the descriptor
        let norm: f64 = descriptor.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut descriptor {
                *v /= norm;
            }
        }

        // Clamp values at 0.2 for robustness against illumination changes
        let clamp_val = 0.2;
        for v in &mut descriptor {
            if *v > clamp_val {
                *v = clamp_val;
            }
        }

        // Re-normalize after clamping
        let norm2: f64 = descriptor.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm2 > 1e-10 {
            for v in &mut descriptor {
                *v /= norm2;
            }
        }

        descriptors.push(FeatureDescriptor {
            keypoint: kp.clone(),
            descriptor,
        });
    }

    Ok(descriptors)
}

/// Computes the L2 (Euclidean) distance between two descriptor vectors.
///
/// If the vectors have different lengths, only the overlapping portion
/// is compared, and the remaining elements contribute as if the shorter
/// vector were zero-padded.
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let max_len = a.len().max(b.len());
    let mut sum = 0.0;
    for i in 0..max_len {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        let diff = va - vb;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Performs brute-force feature matching between two sets of descriptors.
///
/// For each query descriptor, finds the closest train descriptor using
/// L2 (Euclidean) distance. Only matches with distance below `max_distance`
/// are included in the result.
///
/// # Arguments
/// * `descriptors1` - Query descriptors (first image)
/// * `descriptors2` - Train (reference) descriptors (second image)
/// * `max_distance` - Maximum L2 distance to accept a match
///
/// # Returns
/// A vector of matches sorted by distance (ascending)
pub fn brute_force_match(
    descriptors1: &[FeatureDescriptor],
    descriptors2: &[FeatureDescriptor],
    max_distance: f64,
) -> Vec<FeatureMatch> {
    if descriptors1.is_empty() || descriptors2.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();

    for (qi, q) in descriptors1.iter().enumerate() {
        let mut best_dist = f64::INFINITY;
        let mut best_idx = 0_usize;

        for (ti, t) in descriptors2.iter().enumerate() {
            let dist = l2_distance(&q.descriptor, &t.descriptor);
            if dist < best_dist {
                best_dist = dist;
                best_idx = ti;
            }
        }

        if best_dist <= max_distance && best_dist.is_finite() {
            matches.push(FeatureMatch {
                query_idx: qi,
                train_idx: best_idx,
                distance: best_dist,
            });
        }
    }

    // Sort by distance (ascending)
    matches.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test image with a single bright corner (L-shaped pattern).
    fn make_corner_image(size: usize) -> Image {
        let mut data = vec![0.0; size * size];
        let center = size / 2;
        // Create an L-shaped bright pattern: bottom-left quadrant
        for row in center..size {
            for col in 0..=center {
                data[row * size + col] = 1.0;
            }
        }
        Image::from_grayscale(size, size, &data).expect("test: image creation should succeed")
    }

    /// Create a checkerboard image for corner detection tests.
    fn make_checkerboard(size: usize, cell_size: usize) -> Image {
        let mut data = vec![0.0; size * size];
        for row in 0..size {
            for col in 0..size {
                let cell_row = row / cell_size;
                let cell_col = col / cell_size;
                if (cell_row + cell_col).is_multiple_of(2) {
                    data[row * size + col] = 1.0;
                }
            }
        }
        Image::from_grayscale(size, size, &data).expect("test: image creation should succeed")
    }

    /// Create an image with a gradient (brightness increases left to right).
    fn make_gradient_image(size: usize) -> Image {
        let mut data = vec![0.0; size * size];
        for row in 0..size {
            for col in 0..size {
                data[row * size + col] = col as f64 / (size - 1) as f64;
            }
        }
        Image::from_grayscale(size, size, &data).expect("test: image creation should succeed")
    }

    #[test]
    fn test_harris_corner_detect_on_corner_image() {
        let img = make_corner_image(32);
        let keypoints = harris_corner_detect(&img, 3, 0.04, 0.0)
            .expect("test: Harris corner detection should succeed");
        assert!(
            !keypoints.is_empty(),
            "Harris should detect corners in L-shaped image"
        );
        // Keypoints should be sorted by descending response
        for window in keypoints.windows(2) {
            assert!(
                window[0].response >= window[1].response,
                "Keypoints should be sorted by descending response"
            );
        }
    }

    #[test]
    fn test_harris_on_constant_image() {
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let keypoints = harris_corner_detect(&img, 3, 0.04, 0.001)
            .expect("test: Harris should succeed on constant");
        assert!(
            keypoints.is_empty(),
            "Harris should not detect corners in constant image"
        );
    }

    #[test]
    fn test_harris_on_checkerboard() {
        let img = make_checkerboard(32, 8);
        let keypoints = harris_corner_detect(&img, 3, 0.04, 0.0)
            .expect("test: Harris should succeed on checkerboard");
        assert!(
            keypoints.len() > 2,
            "Harris should detect multiple corners on checkerboard: got {}",
            keypoints.len()
        );
    }

    #[test]
    fn test_harris_requires_grayscale() {
        let data = vec![0.5; 8 * 8 * 3];
        let img = Image::from_rgb(8, 8, &data).expect("test: RGB image creation should succeed");
        let result = harris_corner_detect(&img, 3, 0.04, 0.0);
        assert!(result.is_err(), "Harris should reject non-grayscale images");
    }

    #[test]
    fn test_fast_corner_detect_on_checkerboard() {
        // Use larger cells (16px) so the Bresenham circle of radius 3 can fit
        // mostly within one cell's boundary region.  On a checkerboard, a straight
        // edge between two cells produces ~7 contiguous circle pixels on one side,
        // so n_consecutive=5 is appropriate (standard FAST-9 would require 9 which
        // is geometrically impossible on a checkerboard).
        let img = make_checkerboard(64, 16);
        let keypoints =
            fast_corner_detect(&img, 0.3, 5).expect("test: FAST should succeed on checkerboard");
        assert!(
            !keypoints.is_empty(),
            "FAST should detect corners on checkerboard"
        );
    }

    #[test]
    fn test_fast_corner_detect_on_constant() {
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let keypoints =
            fast_corner_detect(&img, 0.05, 9).expect("test: FAST should succeed on constant");
        assert!(
            keypoints.is_empty(),
            "FAST should not detect corners in constant image"
        );
    }

    #[test]
    fn test_fast_invalid_parameters() {
        let img = Image::zeros_grayscale(16, 16);
        let result = fast_corner_detect(&img, 0.05, 0);
        assert!(result.is_err(), "FAST should reject n_consecutive=0");
        let result = fast_corner_detect(&img, 0.05, 17);
        assert!(result.is_err(), "FAST should reject n_consecutive=17");
        let result = fast_corner_detect(&img, -0.1, 9);
        assert!(result.is_err(), "FAST should reject negative threshold");
    }

    #[test]
    fn test_non_maximum_suppression_basic() {
        let keypoints = vec![
            Keypoint::new(10, 10, 100.0),
            Keypoint::new(12, 12, 80.0), // within radius of first
            Keypoint::new(50, 50, 90.0), // far away
            Keypoint::new(51, 51, 70.0), // within radius of third
        ];

        let suppressed = non_maximum_suppression(&keypoints, 5.0);
        assert_eq!(
            suppressed.len(),
            2,
            "NMS should keep only 2 keypoints: got {}",
            suppressed.len()
        );
        assert!((suppressed[0].response - 100.0).abs() < 1e-10);
        assert!((suppressed[1].response - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_maximum_suppression_empty() {
        let keypoints: Vec<Keypoint> = vec![];
        let suppressed = non_maximum_suppression(&keypoints, 5.0);
        assert!(suppressed.is_empty());
    }

    #[test]
    fn test_non_maximum_suppression_single() {
        let keypoints = vec![Keypoint::new(5, 5, 42.0)];
        let suppressed = non_maximum_suppression(&keypoints, 10.0);
        assert_eq!(suppressed.len(), 1);
        assert!((suppressed[0].response - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_descriptor_on_checkerboard() {
        let img = make_checkerboard(64, 8);
        let keypoints = vec![Keypoint::new(32, 32, 1.0), Keypoint::new(24, 24, 0.8)];
        let descs = simple_descriptor(&img, &keypoints, 16)
            .expect("test: descriptor computation should succeed");
        // Both keypoints should be far enough from border
        assert!(
            !descs.is_empty(),
            "Descriptor extraction should produce results for interior keypoints"
        );
        // Descriptor length should be 4*4*8 = 128
        for desc in &descs {
            assert_eq!(
                desc.descriptor.len(),
                128,
                "Descriptor should be 128-dimensional"
            );
            // Descriptor should be normalized (unit length or zero)
            let norm: f64 = desc.descriptor.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01 || norm < 1e-10,
                "Descriptor should be normalized: norm = {}",
                norm
            );
        }
    }

    #[test]
    fn test_simple_descriptor_skips_border_keypoints() {
        let img = Image::zeros_grayscale(32, 32);
        let keypoints = vec![
            Keypoint::new(0, 0, 1.0),   // too close to top-left
            Keypoint::new(31, 31, 1.0), // too close to bottom-right
            Keypoint::new(16, 16, 1.0), // center, should work
        ];
        let descs =
            simple_descriptor(&img, &keypoints, 16).expect("test: descriptor should handle border");
        // Only the center keypoint should produce a descriptor
        assert_eq!(
            descs.len(),
            1,
            "Only interior keypoints should produce descriptors: got {}",
            descs.len()
        );
    }

    #[test]
    fn test_brute_force_match_identical() {
        let desc1 = FeatureDescriptor {
            keypoint: Keypoint::new(10, 10, 1.0),
            descriptor: vec![1.0, 0.0, 0.0, 0.0],
        };
        let desc2 = FeatureDescriptor {
            keypoint: Keypoint::new(20, 20, 1.0),
            descriptor: vec![1.0, 0.0, 0.0, 0.0],
        };

        let matches = brute_force_match(&[desc1], &[desc2], 1.0);
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0].distance < 1e-10,
            "Identical descriptors should match with distance ~0: got {}",
            matches[0].distance
        );
    }

    #[test]
    fn test_brute_force_match_max_distance_filter() {
        let desc1 = FeatureDescriptor {
            keypoint: Keypoint::new(10, 10, 1.0),
            descriptor: vec![1.0, 0.0, 0.0, 0.0],
        };
        let desc2 = FeatureDescriptor {
            keypoint: Keypoint::new(20, 20, 1.0),
            descriptor: vec![0.0, 1.0, 0.0, 0.0],
        };

        // L2 distance between [1,0,0,0] and [0,1,0,0] is sqrt(2) ~ 1.414
        let matches_tight = brute_force_match(
            std::slice::from_ref(&desc1),
            std::slice::from_ref(&desc2),
            1.0,
        );
        assert!(
            matches_tight.is_empty(),
            "Should reject match with distance > max_distance"
        );

        let matches_loose = brute_force_match(&[desc1], &[desc2], 2.0);
        assert_eq!(
            matches_loose.len(),
            1,
            "Should accept match within max_distance"
        );
    }

    #[test]
    fn test_brute_force_match_empty() {
        let matches = brute_force_match(&[], &[], 1.0);
        assert!(matches.is_empty());

        let desc = FeatureDescriptor {
            keypoint: Keypoint::new(5, 5, 1.0),
            descriptor: vec![1.0, 0.0],
        };
        let matches2 = brute_force_match(std::slice::from_ref(&desc), &[], 1.0);
        assert!(matches2.is_empty());
        let matches3 = brute_force_match(&[], &[desc], 1.0);
        assert!(matches3.is_empty());
    }

    #[test]
    fn test_brute_force_match_best_selection() {
        let query = vec![FeatureDescriptor {
            keypoint: Keypoint::new(10, 10, 1.0),
            descriptor: vec![1.0, 0.0, 0.0],
        }];
        let train = vec![
            FeatureDescriptor {
                keypoint: Keypoint::new(20, 20, 1.0),
                descriptor: vec![0.0, 1.0, 0.0], // dist = sqrt(2) ~ 1.414
            },
            FeatureDescriptor {
                keypoint: Keypoint::new(30, 30, 1.0),
                descriptor: vec![0.9, 0.1, 0.0], // dist ~ 0.1414
            },
        ];

        let matches = brute_force_match(&query, &train, 2.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].train_idx, 1,
            "Should select closest descriptor (index 1)"
        );
    }

    #[test]
    fn test_l2_distance_values() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = l2_distance(&a, &b);
        assert!(
            (dist - std::f64::consts::SQRT_2).abs() < 1e-10,
            "L2 distance between orthogonal unit vectors should be sqrt(2): got {}",
            dist
        );

        let c = vec![1.0, 2.0, 3.0];
        let d = vec![1.0, 2.0, 3.0];
        assert!(
            l2_distance(&c, &d) < 1e-10,
            "Identical vectors should have distance 0"
        );

        let e = vec![0.0];
        let f = vec![3.0, 4.0];
        let dist2 = l2_distance(&e, &f);
        // sqrt(3^2 + 4^2) = 5
        assert!(
            (dist2 - 5.0).abs() < 1e-10,
            "Zero-padded distance: got {}",
            dist2
        );
    }

    #[test]
    fn test_has_n_contiguous_wrap_around() {
        // Test wrap-around: 3 true at end + start of circle
        let mut arr = [false; 32];
        // Set positions 14, 15 and also 0, 16, 17 (duplicated circle)
        arr[14] = true;
        arr[15] = true;
        arr[16] = true; // duplicate of position 0
        arr[17] = true; // duplicate of position 1
        arr[0] = true; // position 0
        arr[1] = true; // position 1
        arr[30] = true; // duplicate of position 14
        arr[31] = true; // duplicate of position 15
        assert!(has_n_contiguous(&arr, 4));
        assert!(!has_n_contiguous(&arr, 5));
    }

    #[test]
    fn test_keypoint_constructors() {
        let kp1 = Keypoint::new(10, 20, 5.0);
        assert_eq!(kp1.row, 10);
        assert_eq!(kp1.col, 20);
        assert!((kp1.response - 5.0).abs() < 1e-10);
        assert!((kp1.scale - 1.0).abs() < 1e-10);
        assert!((kp1.orientation).abs() < 1e-10);

        let kp2 = Keypoint::with_full(5, 15, 3.0, 2.0, 1.5);
        assert_eq!(kp2.row, 5);
        assert_eq!(kp2.col, 15);
        assert!((kp2.response - 3.0).abs() < 1e-10);
        assert!((kp2.scale - 2.0).abs() < 1e-10);
        assert!((kp2.orientation - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_descriptor_on_gradient_image() {
        let img = make_gradient_image(64);
        let keypoints = vec![Keypoint::new(32, 32, 1.0)];
        let descs = simple_descriptor(&img, &keypoints, 16)
            .expect("test: descriptor on gradient image should succeed");
        assert_eq!(descs.len(), 1);
        // Gradient image should produce non-trivial descriptors
        let nonzero_count = descs[0]
            .descriptor
            .iter()
            .filter(|&&v| v.abs() > 1e-6)
            .count();
        assert!(
            nonzero_count > 0,
            "Gradient image should produce non-zero descriptor entries"
        );
    }
}

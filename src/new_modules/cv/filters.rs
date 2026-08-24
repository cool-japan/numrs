//! Image Filtering Operations
//!
//! Provides fundamental image filtering operations including convolution,
//! blurring, edge detection, and edge-preserving smoothing.
//!
//! # Operations
//!
//! - **Convolution**: General 2D convolution with configurable border handling
//! - **Gaussian blur**: Smoothing with Gaussian kernel (configurable sigma)
//! - **Box blur / mean filter**: Uniform averaging filter
//! - **Median filter**: Non-linear noise reduction
//! - **Sobel operator**: First-order gradient estimation (x and y)
//! - **Laplacian**: Second-order derivative for edge detection
//! - **Bilateral filter**: Edge-preserving smoothing
//! - **Canny edge detection**: Multi-stage edge detection pipeline
//!
//! # SCIRS2 Policy
//!
//! All implementations use `scirs2_core::ndarray` for array operations
//! and follow the pure Rust requirement.

use super::{BorderMode, ColorSpace, CvError, Image};
use crate::array::Array;
use crate::error::NumRs2Error;

/// Fetches a pixel value with border handling.
///
/// Returns the pixel value at `(row, col)` in the image, applying the
/// specified border mode if the coordinates are outside the image bounds.
fn fetch_pixel(img: &Image, row: isize, col: isize, border: BorderMode, pad_value: f64) -> f64 {
    let h = img.height() as isize;
    let w = img.width() as isize;

    let (r, c) = match border {
        BorderMode::Constant => {
            if row < 0 || row >= h || col < 0 || col >= w {
                return pad_value;
            }
            (row as usize, col as usize)
        }
        BorderMode::Reflect => {
            let r = reflect_index(row, h);
            let c = reflect_index(col, w);
            (r, c)
        }
        BorderMode::Replicate => {
            let r = row.clamp(0, h - 1) as usize;
            let c = col.clamp(0, w - 1) as usize;
            (r, c)
        }
        BorderMode::Wrap => {
            let r = ((row % h) + h) % h;
            let c = ((col % w) + w) % w;
            (r as usize, c as usize)
        }
    };

    img.get_pixel(r, c, 0).unwrap_or(pad_value)
}

/// Reflects an index at the boundary using `dcba|abcd|dcba` pattern.
fn reflect_index(idx: isize, len: isize) -> usize {
    if idx < 0 {
        (-idx - 1).min(len - 1) as usize
    } else if idx >= len {
        let reflected = 2 * len - idx - 1;
        reflected.max(0) as usize
    } else {
        idx as usize
    }
}

/// Performs 2D convolution on a grayscale image with the given kernel.
///
/// The kernel is applied to each pixel position by computing the weighted
/// sum of neighboring pixels. The kernel should be a 2D array with odd
/// dimensions.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `kernel` - 2D convolution kernel (must have odd dimensions)
/// * `border` - Border handling mode
///
/// # Returns
/// A new image with the convolution result
///
/// # Errors
/// Returns error if the kernel has even dimensions or the image is not grayscale
pub fn convolve2d(
    img: &Image,
    kernel: &Array<f64>,
    border: BorderMode,
) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }

    let k_shape = kernel.shape();
    if k_shape.len() != 2 {
        return Err(CvError::InvalidParameter("Kernel must be 2D".to_string()).into());
    }

    let kh = k_shape[0];
    let kw = k_shape[1];

    if kh.is_multiple_of(2) || kw.is_multiple_of(2) {
        return Err(CvError::InvalidKernelSize(kh).into());
    }

    let half_h = (kh / 2) as isize;
    let half_w = (kw / 2) as isize;
    let h = img.height();
    let w = img.width();

    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0;
            for ki in 0..kh {
                for kj in 0..kw {
                    let img_row = row as isize + ki as isize - half_h;
                    let img_col = col as isize + kj as isize - half_w;
                    let pixel = fetch_pixel(img, img_row, img_col, border, 0.0);
                    let k_val = kernel.get(&[ki, kj]).map_err(|e| {
                        NumRs2Error::ComputationError(format!("Kernel access: {}", e))
                    })?;
                    sum += pixel * k_val;
                }
            }
            result.set_pixel(row, col, 0, sum).map_err(|e| {
                NumRs2Error::ComputationError(format!("Setting convolution result: {}", e))
            })?;
        }
    }

    Ok(result)
}

/// Generates a 1D Gaussian kernel.
///
/// # Arguments
/// * `size` - Kernel size (must be odd)
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
/// A normalized 1D Gaussian kernel as a vector
fn gaussian_kernel_1d(size: usize, sigma: f64) -> Result<Vec<f64>, NumRs2Error> {
    if size.is_multiple_of(2) || size == 0 {
        return Err(CvError::InvalidKernelSize(size).into());
    }
    if sigma <= 0.0 {
        return Err(CvError::InvalidParameter("Sigma must be positive".to_string()).into());
    }

    let half = (size / 2) as isize;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel = Vec::with_capacity(size);
    let mut total = 0.0;

    for i in -half..=half {
        let val = (-(i * i) as f64 / two_sigma_sq).exp();
        kernel.push(val);
        total += val;
    }

    // Normalize
    for v in &mut kernel {
        *v /= total;
    }

    Ok(kernel)
}

/// Generates a 2D Gaussian kernel.
///
/// # Arguments
/// * `size` - Kernel size (must be odd)
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
/// A normalized 2D Gaussian kernel as an Array
fn gaussian_kernel_2d(size: usize, sigma: f64) -> Result<Array<f64>, NumRs2Error> {
    if size.is_multiple_of(2) || size == 0 {
        return Err(CvError::InvalidKernelSize(size).into());
    }
    if sigma <= 0.0 {
        return Err(CvError::InvalidParameter("Sigma must be positive".to_string()).into());
    }

    let half = (size / 2) as isize;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut data = Vec::with_capacity(size * size);
    let mut total = 0.0;

    for i in -half..=half {
        for j in -half..=half {
            let val = (-((i * i + j * j) as f64) / two_sigma_sq).exp();
            data.push(val);
            total += val;
        }
    }

    // Normalize
    for v in &mut data {
        *v /= total;
    }

    Array::from_vec_shape(data, &[size, size])
}

/// Applies Gaussian blur to a grayscale image.
///
/// The Gaussian blur is a low-pass filter that smooths the image by
/// convolving with a Gaussian kernel. This reduces high-frequency noise
/// while preserving low-frequency structure.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `kernel_size` - Size of the Gaussian kernel (must be odd and positive)
/// * `sigma` - Standard deviation of the Gaussian distribution
///
/// # Returns
/// A new image with Gaussian blur applied
///
/// # Errors
/// Returns error if kernel size is even or sigma is non-positive
pub fn gaussian_blur(img: &Image, kernel_size: usize, sigma: f64) -> Result<Image, NumRs2Error> {
    let kernel = gaussian_kernel_2d(kernel_size, sigma)?;
    convolve2d(img, &kernel, BorderMode::Reflect)
}

/// Applies the Sobel operator in the x-direction (horizontal gradient).
///
/// The Sobel operator estimates the image gradient, highlighting
/// vertical edges (intensity changes in the horizontal direction).
///
/// The 3x3 Sobel kernel for x-gradient is:
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
///
/// # Arguments
/// * `img` - Input grayscale image
///
/// # Returns
/// A new image containing the horizontal gradient
pub fn sobel_x(img: &Image) -> Result<Image, NumRs2Error> {
    let kernel_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let kernel = Array::from_vec_shape(kernel_data, &[3, 3])?;
    convolve2d(img, &kernel, BorderMode::Reflect)
}

/// Applies the Sobel operator in the y-direction (vertical gradient).
///
/// The Sobel operator estimates the image gradient, highlighting
/// horizontal edges (intensity changes in the vertical direction).
///
/// The 3x3 Sobel kernel for y-gradient is:
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
///
/// # Arguments
/// * `img` - Input grayscale image
///
/// # Returns
/// A new image containing the vertical gradient
pub fn sobel_y(img: &Image) -> Result<Image, NumRs2Error> {
    let kernel_data = vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let kernel = Array::from_vec_shape(kernel_data, &[3, 3])?;
    convolve2d(img, &kernel, BorderMode::Reflect)
}

/// Applies the Laplacian filter for edge detection.
///
/// The Laplacian is a second-order derivative operator that detects
/// regions of rapid intensity change. It highlights edges and is
/// often used for sharpening.
///
/// The 3x3 Laplacian kernel is:
/// ```text
/// [0  1  0]
/// [1 -4  1]
/// [0  1  0]
/// ```
///
/// # Arguments
/// * `img` - Input grayscale image
///
/// # Returns
/// A new image containing the Laplacian response
pub fn laplacian_filter(img: &Image) -> Result<Image, NumRs2Error> {
    let kernel_data = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let kernel = Array::from_vec_shape(kernel_data, &[3, 3])?;
    convolve2d(img, &kernel, BorderMode::Reflect)
}

/// Applies a box blur (mean filter) to a grayscale image.
///
/// Each pixel is replaced by the average of its neighbors within
/// the specified kernel size. This is a simple low-pass filter.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `kernel_size` - Size of the averaging kernel (must be odd)
///
/// # Returns
/// A new image with the box blur applied
///
/// # Errors
/// Returns error if kernel size is even
pub fn box_blur(img: &Image, kernel_size: usize) -> Result<Image, NumRs2Error> {
    if kernel_size.is_multiple_of(2) || kernel_size == 0 {
        return Err(CvError::InvalidKernelSize(kernel_size).into());
    }

    let n = (kernel_size * kernel_size) as f64;
    let data = vec![1.0 / n; kernel_size * kernel_size];
    let kernel = Array::from_vec_shape(data, &[kernel_size, kernel_size])?;
    convolve2d(img, &kernel, BorderMode::Reflect)
}

/// Applies a median filter to a grayscale image.
///
/// The median filter is a non-linear filter that replaces each pixel
/// with the median of its neighbors. It is highly effective at removing
/// salt-and-pepper noise while preserving edges.
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `kernel_size` - Size of the median filter window (must be odd)
///
/// # Returns
/// A new image with the median filter applied
///
/// # Errors
/// Returns error if kernel size is even or image is not grayscale
pub fn median_filter(img: &Image, kernel_size: usize) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if kernel_size.is_multiple_of(2) || kernel_size == 0 {
        return Err(CvError::InvalidKernelSize(kernel_size).into());
    }

    let h = img.height();
    let w = img.width();
    let half = (kernel_size / 2) as isize;
    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let mut values = Vec::with_capacity(kernel_size * kernel_size);
            for ki in -half..=half {
                for kj in -half..=half {
                    let pixel = fetch_pixel(
                        img,
                        row as isize + ki,
                        col as isize + kj,
                        BorderMode::Reflect,
                        0.0,
                    );
                    values.push(pixel);
                }
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = values[values.len() / 2];
            result.set_pixel(row, col, 0, median).map_err(|e| {
                NumRs2Error::ComputationError(format!("Setting median result: {}", e))
            })?;
        }
    }

    Ok(result)
}

/// Applies a bilateral filter to a grayscale image.
///
/// The bilateral filter is an edge-preserving smoothing filter. It combines
/// spatial proximity with intensity similarity, so that nearby pixels with
/// similar intensities are averaged together while edges are preserved.
///
/// The filter weight for pixel `q` relative to center pixel `p` is:
///
/// `w(p, q) = exp(-||p-q||^2 / (2*sigma_s^2)) * exp(-|I(p)-I(q)|^2 / (2*sigma_r^2))`
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `kernel_size` - Size of the filter window (must be odd)
/// * `sigma_spatial` - Standard deviation for spatial distance weighting
/// * `sigma_range` - Standard deviation for intensity difference weighting
///
/// # Returns
/// A new image with bilateral filtering applied
///
/// # Errors
/// Returns error if parameters are invalid
pub fn bilateral_filter(
    img: &Image,
    kernel_size: usize,
    sigma_spatial: f64,
    sigma_range: f64,
) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if kernel_size.is_multiple_of(2) || kernel_size == 0 {
        return Err(CvError::InvalidKernelSize(kernel_size).into());
    }
    if sigma_spatial <= 0.0 || sigma_range <= 0.0 {
        return Err(CvError::InvalidParameter("Sigma values must be positive".to_string()).into());
    }

    let h = img.height();
    let w = img.width();
    let half = (kernel_size / 2) as isize;
    let two_sigma_s_sq = 2.0 * sigma_spatial * sigma_spatial;
    let two_sigma_r_sq = 2.0 * sigma_range * sigma_range;

    let mut result = Image::zeros_grayscale(w, h);

    for row in 0..h {
        for col in 0..w {
            let center = fetch_pixel(img, row as isize, col as isize, BorderMode::Reflect, 0.0);
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for ki in -half..=half {
                for kj in -half..=half {
                    let pixel = fetch_pixel(
                        img,
                        row as isize + ki,
                        col as isize + kj,
                        BorderMode::Reflect,
                        0.0,
                    );

                    // Spatial weight
                    let spatial_dist_sq = (ki * ki + kj * kj) as f64;
                    let spatial_weight = (-spatial_dist_sq / two_sigma_s_sq).exp();

                    // Range (intensity) weight
                    let intensity_diff = pixel - center;
                    let range_weight = (-(intensity_diff * intensity_diff) / two_sigma_r_sq).exp();

                    let weight = spatial_weight * range_weight;
                    weighted_sum += pixel * weight;
                    weight_sum += weight;
                }
            }

            let filtered_val = if weight_sum > 1e-15 {
                weighted_sum / weight_sum
            } else {
                center
            };

            result.set_pixel(row, col, 0, filtered_val).map_err(|e| {
                NumRs2Error::ComputationError(format!("Setting bilateral result: {}", e))
            })?;
        }
    }

    Ok(result)
}

/// Applies Canny edge detection to a grayscale image.
///
/// The Canny edge detector is a multi-stage algorithm:
/// 1. Apply Gaussian blur to reduce noise
/// 2. Compute gradient magnitude and direction using Sobel operators
/// 3. Apply non-maximum suppression to thin edges
/// 4. Apply double threshold and hysteresis to detect strong/weak edges
///
/// # Arguments
/// * `img` - Input grayscale image
/// * `low_threshold` - Lower threshold for hysteresis (e.g., 0.05)
/// * `high_threshold` - Upper threshold for hysteresis (e.g., 0.15)
/// * `sigma` - Gaussian blur sigma for noise reduction (e.g., 1.4)
///
/// # Returns
/// A binary edge map where edges have value 1.0 and non-edges have value 0.0
///
/// # Errors
/// Returns error if parameters are invalid or image is not grayscale
pub fn canny_edge_detect(
    img: &Image,
    low_threshold: f64,
    high_threshold: f64,
    sigma: f64,
) -> Result<Image, NumRs2Error> {
    if img.color_space() != ColorSpace::Grayscale {
        return Err(CvError::RequiresGrayscale.into());
    }
    if low_threshold < 0.0 || high_threshold < 0.0 || low_threshold > high_threshold {
        return Err(CvError::InvalidParameter(
            "Thresholds must be non-negative and low <= high".to_string(),
        )
        .into());
    }

    let h = img.height();
    let w = img.width();

    // Step 1: Gaussian blur
    let blurred = gaussian_blur(img, 5, sigma)?;

    // Step 2: Compute gradients
    let gx = sobel_x(&blurred)?;
    let gy = sobel_y(&blurred)?;

    // Compute gradient magnitude and direction
    let mut magnitude = Image::zeros_grayscale(w, h);
    let mut direction = vec![0.0_f64; h * w]; // angle in radians

    let mut max_mag = 0.0_f64;
    for row in 0..h {
        for col in 0..w {
            let dx = gx
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("Gradient x: {}", e)))?;
            let dy = gy
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("Gradient y: {}", e)))?;
            let mag = (dx * dx + dy * dy).sqrt();
            if mag > max_mag {
                max_mag = mag;
            }
            magnitude
                .set_pixel(row, col, 0, mag)
                .map_err(|e| NumRs2Error::ComputationError(format!("Setting magnitude: {}", e)))?;
            direction[row * w + col] = dy.atan2(dx);
        }
    }

    // Normalize magnitude to [0, 1]
    if max_mag > 1e-15 {
        for row in 0..h {
            for col in 0..w {
                let v = magnitude
                    .get_pixel(row, col, 0)
                    .map_err(|e| NumRs2Error::ComputationError(format!("Mag read: {}", e)))?;
                magnitude
                    .set_pixel(row, col, 0, v / max_mag)
                    .map_err(|e| NumRs2Error::ComputationError(format!("Mag normalize: {}", e)))?;
            }
        }
    }

    // Step 3: Non-maximum suppression
    let mut suppressed = Image::zeros_grayscale(w, h);

    for row in 1..(h.saturating_sub(1)) {
        for col in 1..(w.saturating_sub(1)) {
            let angle = direction[row * w + col];
            let mag = magnitude
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("NMS mag read: {}", e)))?;

            // Quantize angle to 4 directions: 0, 45, 90, 135 degrees
            let angle_deg = angle.to_degrees();
            let normalized = ((angle_deg % 180.0) + 180.0) % 180.0;

            let (n1, n2) = if !(22.5..157.5).contains(&normalized) {
                // Horizontal (0 degrees): compare with left and right
                (
                    magnitude
                        .get_pixel(row, col.wrapping_sub(1), 0)
                        .unwrap_or(0.0),
                    magnitude.get_pixel(row, col + 1, 0).unwrap_or(0.0),
                )
            } else if normalized < 67.5 {
                // Diagonal (45 degrees): compare with top-right and bottom-left
                (
                    magnitude
                        .get_pixel(row.wrapping_sub(1), col + 1, 0)
                        .unwrap_or(0.0),
                    magnitude
                        .get_pixel(row + 1, col.wrapping_sub(1), 0)
                        .unwrap_or(0.0),
                )
            } else if normalized < 112.5 {
                // Vertical (90 degrees): compare with top and bottom
                (
                    magnitude
                        .get_pixel(row.wrapping_sub(1), col, 0)
                        .unwrap_or(0.0),
                    magnitude.get_pixel(row + 1, col, 0).unwrap_or(0.0),
                )
            } else {
                // Diagonal (135 degrees): compare with top-left and bottom-right
                (
                    magnitude
                        .get_pixel(row.wrapping_sub(1), col.wrapping_sub(1), 0)
                        .unwrap_or(0.0),
                    magnitude.get_pixel(row + 1, col + 1, 0).unwrap_or(0.0),
                )
            };

            if mag >= n1 && mag >= n2 {
                suppressed
                    .set_pixel(row, col, 0, mag)
                    .map_err(|e| NumRs2Error::ComputationError(format!("NMS set: {}", e)))?;
            }
        }
    }

    // Step 4: Double threshold and hysteresis
    // Mark pixels as strong (1.0), weak (0.5), or none (0.0)
    let mut edges = Image::zeros_grayscale(w, h);
    let strong = 1.0;
    let weak = 0.5;

    for row in 0..h {
        for col in 0..w {
            let val = suppressed
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("Threshold read: {}", e)))?;
            if val >= high_threshold {
                edges.set_pixel(row, col, 0, strong).map_err(|e| {
                    NumRs2Error::ComputationError(format!("Strong edge set: {}", e))
                })?;
            } else if val >= low_threshold {
                edges
                    .set_pixel(row, col, 0, weak)
                    .map_err(|e| NumRs2Error::ComputationError(format!("Weak edge set: {}", e)))?;
            }
        }
    }

    // Hysteresis: weak edges connected to strong edges become strong
    let mut changed = true;
    while changed {
        changed = false;
        for row in 1..(h.saturating_sub(1)) {
            for col in 1..(w.saturating_sub(1)) {
                let val = edges.get_pixel(row, col, 0).map_err(|e| {
                    NumRs2Error::ComputationError(format!("Hysteresis read: {}", e))
                })?;
                if (val - weak).abs() < 1e-10 {
                    // Check 8-connected neighbors for strong edges
                    let mut has_strong_neighbor = false;
                    for dr in -1_isize..=1 {
                        for dc in -1_isize..=1 {
                            if dr == 0 && dc == 0 {
                                continue;
                            }
                            let nr = (row as isize + dr) as usize;
                            let nc = (col as isize + dc) as usize;
                            let neighbor_val = edges.get_pixel(nr, nc, 0).unwrap_or(0.0);
                            if (neighbor_val - strong).abs() < 1e-10 {
                                has_strong_neighbor = true;
                                break;
                            }
                        }
                        if has_strong_neighbor {
                            break;
                        }
                    }
                    if has_strong_neighbor {
                        edges.set_pixel(row, col, 0, strong).map_err(|e| {
                            NumRs2Error::ComputationError(format!("Hysteresis promote: {}", e))
                        })?;
                        changed = true;
                    }
                }
            }
        }
    }

    // Final pass: remove remaining weak edges
    for row in 0..h {
        for col in 0..w {
            let val = edges
                .get_pixel(row, col, 0)
                .map_err(|e| NumRs2Error::ComputationError(format!("Final cleanup read: {}", e)))?;
            if (val - strong).abs() > 1e-10 {
                edges.set_pixel(row, col, 0, 0.0).map_err(|e| {
                    NumRs2Error::ComputationError(format!("Final cleanup set: {}", e))
                })?;
            }
        }
    }

    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple test image with known values.
    fn make_test_image(w: usize, h: usize) -> Image {
        let mut data = vec![0.0; w * h];
        for row in 0..h {
            for col in 0..w {
                data[row * w + col] = (row * w + col) as f64 / (w * h) as f64;
            }
        }
        Image::from_grayscale(w, h, &data).expect("test: image creation should succeed")
    }

    /// Helper to create an image with a bright rectangle in the center.
    fn make_edge_image(w: usize, h: usize) -> Image {
        let mut data = vec![0.0; w * h];
        let margin_h = h / 4;
        let margin_w = w / 4;
        for row in margin_h..(h - margin_h) {
            for col in margin_w..(w - margin_w) {
                data[row * w + col] = 1.0;
            }
        }
        Image::from_grayscale(w, h, &data).expect("test: image creation should succeed")
    }

    #[test]
    fn test_convolve2d_identity() {
        let img = make_test_image(8, 8);
        // Identity kernel: [0 0 0; 0 1 0; 0 0 0]
        let kernel_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let kernel = Array::from_vec(kernel_data).reshape(&[3, 3]);
        let result = convolve2d(&img, &kernel, BorderMode::Constant)
            .expect("test: convolution should succeed");
        // For interior pixels, the identity kernel should reproduce the original
        for row in 1..7 {
            for col in 1..7 {
                let orig = img
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                let conv = result
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    (orig - conv).abs() < 1e-10,
                    "Identity convolution should preserve pixel value at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_gaussian_blur_preserves_constant_image() {
        // A constant image should remain unchanged after Gaussian blur
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let blurred = gaussian_blur(&img, 3, 1.0).expect("test: Gaussian blur should succeed");
        // Check center pixels (avoiding border effects)
        for row in 2..14 {
            for col in 2..14 {
                let v = blurred
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    (v - 0.5).abs() < 1e-8,
                    "Gaussian blur should preserve constant image value at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_sobel_detects_vertical_edge() {
        let img = make_edge_image(16, 16);
        let gx = sobel_x(&img).expect("test: Sobel x should succeed");

        // The Sobel x operator should detect vertical edges
        // At the left edge of the bright rectangle, gradients should be positive
        // At the right edge, gradients should be negative
        let edge_col_left = 4; // margin_w = 4
        let center_row = 8;

        let left_grad = gx
            .get_pixel(center_row, edge_col_left, 0)
            .expect("test: pixel read should succeed");
        // The gradient at the left edge should be positive (bright on right, dark on left)
        assert!(
            left_grad > 0.0,
            "Sobel x should detect positive gradient at left edge: got {}",
            left_grad
        );
    }

    #[test]
    fn test_sobel_detects_horizontal_edge() {
        let img = make_edge_image(16, 16);
        let gy = sobel_y(&img).expect("test: Sobel y should succeed");

        let edge_row_top = 4; // margin_h = 4
        let center_col = 8;

        let top_grad = gy
            .get_pixel(edge_row_top, center_col, 0)
            .expect("test: pixel read should succeed");
        // The gradient at the top edge should be positive
        assert!(
            top_grad > 0.0,
            "Sobel y should detect positive gradient at top edge: got {}",
            top_grad
        );
    }

    #[test]
    fn test_laplacian_detects_edges() {
        let img = make_edge_image(16, 16);
        let lap = laplacian_filter(&img).expect("test: Laplacian should succeed");

        // Interior of bright region should be ~0 (constant region)
        let center_val = lap
            .get_pixel(8, 8, 0)
            .expect("test: pixel read should succeed");
        assert!(
            center_val.abs() < 0.01,
            "Laplacian should be near zero in constant region: got {}",
            center_val
        );
    }

    #[test]
    fn test_box_blur() {
        let data = vec![0.5; 8 * 8];
        let img = Image::from_grayscale(8, 8, &data).expect("test: image creation should succeed");
        let blurred = box_blur(&img, 3).expect("test: box blur should succeed");
        // Constant image should remain unchanged
        for row in 1..7 {
            for col in 1..7 {
                let v = blurred
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    (v - 0.5).abs() < 1e-8,
                    "Box blur should preserve constant at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_box_blur_invalid_kernel() {
        let img = Image::zeros_grayscale(8, 8);
        let result = box_blur(&img, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_filter_removes_impulse_noise() {
        // Create an image with a single impulse noise pixel
        let mut data = vec![0.5; 8 * 8];
        data[4 * 8 + 4] = 1.0; // impulse at center
        let img = Image::from_grayscale(8, 8, &data).expect("test: image creation should succeed");
        let filtered = median_filter(&img, 3).expect("test: median filter should succeed");
        // The impulse should be removed
        let center = filtered
            .get_pixel(4, 4, 0)
            .expect("test: pixel read should succeed");
        assert!(
            (center - 0.5).abs() < 1e-8,
            "Median filter should remove impulse noise: got {}",
            center
        );
    }

    #[test]
    fn test_bilateral_filter() {
        let data = vec![0.5; 8 * 8];
        let img = Image::from_grayscale(8, 8, &data).expect("test: image creation should succeed");
        let filtered =
            bilateral_filter(&img, 3, 1.0, 0.1).expect("test: bilateral filter should succeed");
        // Constant image should remain unchanged
        for row in 1..7 {
            for col in 1..7 {
                let v = filtered
                    .get_pixel(row, col, 0)
                    .expect("test: pixel read should succeed");
                assert!(
                    (v - 0.5).abs() < 1e-8,
                    "Bilateral filter should preserve constant at ({}, {}): got {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_bilateral_filter_invalid_params() {
        let img = Image::zeros_grayscale(8, 8);
        let result = bilateral_filter(&img, 3, -1.0, 0.1);
        assert!(result.is_err());
        let result = bilateral_filter(&img, 4, 1.0, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_canny_on_edge_image() {
        let img = make_edge_image(32, 32);
        let edges = canny_edge_detect(&img, 0.05, 0.15, 1.0).expect("test: Canny should succeed");

        // There should be some edge pixels detected
        let edge_data = edges.to_vec();
        let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();
        assert!(
            edge_count > 0,
            "Canny should detect at least some edges in the edge image"
        );
    }

    #[test]
    fn test_canny_on_constant_image() {
        let data = vec![0.5; 16 * 16];
        let img =
            Image::from_grayscale(16, 16, &data).expect("test: image creation should succeed");
        let edges = canny_edge_detect(&img, 0.05, 0.15, 1.0).expect("test: Canny should succeed");

        // No edges should be detected in a constant image
        let edge_data = edges.to_vec();
        let edge_count = edge_data.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(
            edge_count, 0,
            "Canny should detect no edges in a constant image"
        );
    }

    #[test]
    fn test_gaussian_kernel_normalization() {
        let k = gaussian_kernel_1d(5, 1.0).expect("test: kernel generation should succeed");
        let sum: f64 = k.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Gaussian kernel should sum to 1.0: got {}",
            sum
        );
    }

    #[test]
    fn test_reflect_index() {
        assert_eq!(reflect_index(0, 10), 0);
        assert_eq!(reflect_index(9, 10), 9);
        assert_eq!(reflect_index(-1, 10), 0);
        assert_eq!(reflect_index(-2, 10), 1);
        assert_eq!(reflect_index(10, 10), 9);
        assert_eq!(reflect_index(11, 10), 8);
    }
}

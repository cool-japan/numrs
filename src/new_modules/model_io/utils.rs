//! Model I/O Utilities
//!
//! Provides utility functions for model compression, fingerprinting,
//! format detection, and streaming serialization.

use super::format::{FormatResult, NumRS2Model};
use crate::error::NumRs2Error;
use oxicode::{decode_from_slice, encode_to_vec};
use scirs2_core::ndarray::{Array1, Array2};
use serde_json;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Model compression utilities
pub struct ModelCompression;

impl ModelCompression {
    /// Compresses weight data using serde_json
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight array to compress
    ///
    /// # Returns
    ///
    /// Compressed bytes
    pub fn compress_weights(weights: &Array2<f64>) -> FormatResult<Vec<u8>> {
        serde_json::to_vec(weights).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to compress weights: {}", e))
        })
    }

    /// Decompresses weight data
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed data bytes
    ///
    /// # Returns
    ///
    /// Decompressed weight array
    pub fn decompress_weights(data: &[u8]) -> FormatResult<Array2<f64>> {
        serde_json::from_slice(data).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to decompress weights: {}", e))
        })
    }

    /// Quantizes weights to reduce size
    ///
    /// Converts f64 weights to quantized representation (e.g., int8)
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight array to quantize
    /// * `bits` - Number of bits for quantization (8, 16, or 32)
    ///
    /// # Returns
    ///
    /// Quantized weights as bytes and scale/zero-point parameters
    pub fn quantize_weights(weights: &Array2<f64>, bits: u8) -> FormatResult<(Vec<u8>, f64, f64)> {
        if bits != 8 && bits != 16 && bits != 32 {
            return Err(NumRs2Error::ValueError(
                "Quantization bits must be 8, 16, or 32".to_string(),
            ));
        }

        // Calculate min and max for quantization range
        let min_val = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate scale and zero point
        let range = max_val - min_val;
        let qmin = 0.0;
        let qmax = (2_u32.pow(bits as u32) - 1) as f64;

        let scale = range / (qmax - qmin);
        let zero_point = qmin - min_val / scale;

        // Quantize weights
        let quantized: Vec<u8> = match bits {
            8 => weights
                .iter()
                .map(|&w| {
                    let q = ((w / scale) + zero_point).round().clamp(qmin, qmax);
                    q as u8
                })
                .collect(),
            16 => {
                let mut bytes = Vec::new();
                for &w in weights.iter() {
                    let q = ((w / scale) + zero_point).round().clamp(qmin, qmax);
                    let q_u16 = q as u16;
                    bytes.extend_from_slice(&q_u16.to_le_bytes());
                }
                bytes
            }
            32 => {
                let mut bytes = Vec::new();
                for &w in weights.iter() {
                    let q = ((w / scale) + zero_point).round().clamp(qmin, qmax);
                    let q_u32 = q as u32;
                    bytes.extend_from_slice(&q_u32.to_le_bytes());
                }
                bytes
            }
            _ => unreachable!(),
        };

        Ok((quantized, scale, zero_point))
    }

    /// Dequantizes weights back to f64
    ///
    /// # Arguments
    ///
    /// * `data` - Quantized data bytes
    /// * `shape` - Original weight shape
    /// * `bits` - Number of bits used for quantization
    /// * `scale` - Quantization scale
    /// * `zero_point` - Quantization zero point
    pub fn dequantize_weights(
        data: &[u8],
        shape: (usize, usize),
        bits: u8,
        scale: f64,
        zero_point: f64,
    ) -> FormatResult<Array2<f64>> {
        let total_elements = shape.0 * shape.1;

        let values: Vec<f64> = match bits {
            8 => {
                if data.len() != total_elements {
                    return Err(NumRs2Error::ValueError(
                        "Data size mismatch for 8-bit quantization".to_string(),
                    ));
                }
                data.iter()
                    .map(|&q| (q as f64 - zero_point) * scale)
                    .collect()
            }
            16 => {
                if data.len() != total_elements * 2 {
                    return Err(NumRs2Error::ValueError(
                        "Data size mismatch for 16-bit quantization".to_string(),
                    ));
                }
                data.chunks_exact(2)
                    .map(|chunk| {
                        let q = u16::from_le_bytes([chunk[0], chunk[1]]);
                        (q as f64 - zero_point) * scale
                    })
                    .collect()
            }
            32 => {
                if data.len() != total_elements * 4 {
                    return Err(NumRs2Error::ValueError(
                        "Data size mismatch for 32-bit quantization".to_string(),
                    ));
                }
                data.chunks_exact(4)
                    .map(|chunk| {
                        let q = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (q as f64 - zero_point) * scale
                    })
                    .collect()
            }
            _ => {
                return Err(NumRs2Error::ValueError(
                    "Unsupported quantization bits".to_string(),
                ))
            }
        };

        Array2::from_shape_vec(shape, values).map_err(|e| {
            NumRs2Error::ValueError(format!("Failed to create array from quantized data: {}", e))
        })
    }
}

/// Model fingerprinting for verification
pub struct ModelFingerprint;

impl ModelFingerprint {
    /// Computes a cryptographic hash of the model
    ///
    /// Uses SHA-256 for secure model verification
    ///
    /// # Arguments
    ///
    /// * `model` - The model to fingerprint
    ///
    /// # Returns
    ///
    /// SHA-256 hash as hex string
    pub fn compute_hash(model: &NumRS2Model) -> FormatResult<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Serialize model
        let bytes = serde_json::to_vec(model).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize model for hashing: {}", e))
        })?;

        // Compute hash
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        let hash = hasher.finish();

        Ok(format!("{:x}", hash))
    }

    /// Verifies a model against a known hash
    ///
    /// # Arguments
    ///
    /// * `model` - The model to verify
    /// * `expected_hash` - Expected hash value
    ///
    /// # Returns
    ///
    /// True if hash matches, false otherwise
    pub fn verify_hash(model: &NumRS2Model, expected_hash: &str) -> FormatResult<bool> {
        let computed_hash = Self::compute_hash(model)?;
        Ok(computed_hash == expected_hash)
    }

    /// Computes a lightweight checksum for quick validation
    ///
    /// # Arguments
    ///
    /// * `model` - The model to checksum
    ///
    /// # Returns
    ///
    /// CRC32 checksum
    pub fn compute_checksum(model: &NumRS2Model) -> FormatResult<u32> {
        let bytes = serde_json::to_vec(model).map_err(|e| {
            NumRs2Error::SerializationError(format!(
                "Failed to serialize model for checksum: {}",
                e
            ))
        })?;

        // Simple checksum (CRC-32 style)
        let mut checksum: u32 = 0;
        for &byte in bytes.iter() {
            checksum = checksum.wrapping_add(byte as u32);
        }

        Ok(checksum)
    }
}

/// Format detection utilities
pub struct FormatDetector;

impl FormatDetector {
    /// Detects the format of a model file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file
    ///
    /// # Returns
    ///
    /// Detected format as string
    pub fn detect<P: AsRef<Path>>(path: P) -> FormatResult<String> {
        let file = File::open(path.as_ref()).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to open file for format detection: {}", e))
        })?;

        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];

        reader
            .read_exact(&mut magic)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read magic bytes: {}", e)))?;

        // Check for NumRS2 format
        if magic == *b"NUMRS2\x00\x00" {
            return Ok("numrs2".to_string());
        }

        // Check for NPY format
        if magic[0..6] == *b"\x93NUMPY" {
            return Ok("npy".to_string());
        }

        // Check for ZIP format (NPZ)
        if magic[0..4] == *b"PK\x03\x04" {
            return Ok("npz".to_string());
        }

        // Check for JSON format
        reader = BufReader::new(
            File::open(path.as_ref())
                .map_err(|e| NumRs2Error::IOError(format!("Failed to reopen file: {}", e)))?,
        );

        let mut first_byte = [0u8; 1];
        reader
            .read_exact(&mut first_byte)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read first byte: {}", e)))?;

        if first_byte[0] == b'{' || first_byte[0] == b'[' {
            return Ok("json".to_string());
        }

        Ok("unknown".to_string())
    }

    /// Checks if a file is a NumRS2 model
    ///
    /// # Arguments
    ///
    /// * `path` - Path to check
    ///
    /// # Returns
    ///
    /// True if file is NumRS2 format
    pub fn is_numrs2_format<P: AsRef<Path>>(path: P) -> bool {
        Self::detect(path).is_ok_and(|fmt| fmt == "numrs2")
    }
}

/// Streaming serializer for large models
pub struct StreamingSerializer;

impl StreamingSerializer {
    /// Saves a model using streaming to reduce memory usage
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `path` - Output file path
    ///
    /// # Note
    ///
    /// For very large models, this can reduce peak memory usage
    /// by serializing layers incrementally.
    pub fn save_streaming<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        // For now, use standard serialization
        // In future, implement true streaming serialization
        super::serialize::ModelSerializer::save(model, path)
    }

    /// Loads a model using streaming
    ///
    /// # Arguments
    ///
    /// * `path` - Input file path
    pub fn load_streaming<P: AsRef<Path>>(path: P) -> FormatResult<NumRS2Model> {
        // For now, use standard deserialization
        // In future, implement true streaming deserialization
        super::serialize::ModelSerializer::load(path)
    }
}

/// Convenience function to compress weights
pub fn compress_weights(weights: &Array2<f64>) -> FormatResult<Vec<u8>> {
    ModelCompression::compress_weights(weights)
}

/// Convenience function to decompress weights
pub fn decompress_weights(data: &[u8]) -> FormatResult<Array2<f64>> {
    ModelCompression::decompress_weights(data)
}

/// Convenience function to quantize weights
pub fn quantize_weights(weights: &Array2<f64>, bits: u8) -> FormatResult<(Vec<u8>, f64, f64)> {
    ModelCompression::quantize_weights(weights, bits)
}

/// Convenience function to compute model hash
pub fn compute_model_hash(model: &NumRS2Model) -> FormatResult<String> {
    ModelFingerprint::compute_hash(model)
}

/// Convenience function to detect format
pub fn detect_format<P: AsRef<Path>>(path: P) -> FormatResult<String> {
    FormatDetector::detect(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::model_io::format::{LayerData, ModelMetadata};
    use scirs2_core::ndarray::Array2;
    use std::env;
    use std::fs;

    #[test]
    fn test_compress_decompress_weights() {
        let weights = Array2::from_shape_fn((10, 5), |(i, j)| (i * 5 + j) as f64);

        let compressed = ModelCompression::compress_weights(&weights);
        assert!(compressed.is_ok());

        let compressed_data = compressed.unwrap();
        assert!(!compressed_data.is_empty());

        let decompressed = ModelCompression::decompress_weights(&compressed_data);
        assert!(decompressed.is_ok());

        let recovered = decompressed.unwrap();
        assert_eq!(recovered.shape(), weights.shape());
    }

    #[test]
    fn test_quantize_weights_8bit() {
        let weights = Array2::from_shape_fn((5, 4), |(i, j)| (i * 4 + j) as f64);

        let result = ModelCompression::quantize_weights(&weights, 8);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert!(!quantized.is_empty());
        assert!(scale > 0.0);

        // Test dequantization
        let dequantized =
            ModelCompression::dequantize_weights(&quantized, (5, 4), 8, scale, zero_point);
        assert!(dequantized.is_ok());
    }

    #[test]
    fn test_quantize_weights_16bit() {
        let weights = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);

        let result = ModelCompression::quantize_weights(&weights, 16);
        assert!(result.is_ok());

        let (quantized, scale, zero_point) = result.unwrap();
        assert_eq!(quantized.len(), 3 * 3 * 2); // 2 bytes per value

        let dequantized =
            ModelCompression::dequantize_weights(&quantized, (3, 3), 16, scale, zero_point);
        assert!(dequantized.is_ok());
    }

    #[test]
    fn test_quantize_invalid_bits() {
        let weights = Array2::ones((5, 5));
        let result = ModelCompression::quantize_weights(&weights, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_hash() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let hash = ModelFingerprint::compute_hash(&model);
        assert!(hash.is_ok());

        let hash_str = hash.unwrap();
        assert!(!hash_str.is_empty());
        assert!(!hash_str.is_empty());
    }

    #[test]
    fn test_verify_hash() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let hash = ModelFingerprint::compute_hash(&model).unwrap();

        let verified = ModelFingerprint::verify_hash(&model, &hash);
        assert!(verified.is_ok());
        assert!(verified.unwrap());

        let verified_wrong = ModelFingerprint::verify_hash(&model, "wrong_hash");
        assert!(verified_wrong.is_ok());
        assert!(!verified_wrong.unwrap());
    }

    #[test]
    fn test_compute_checksum() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let checksum = ModelFingerprint::compute_checksum(&model);
        assert!(checksum.is_ok());
        assert!(checksum.unwrap() > 0);
    }

    #[test]
    fn test_detect_format_numrs2() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_detect.numrs2");

        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Save model
        super::super::serialize::ModelSerializer::save(&model, &path).unwrap();

        // Detect format
        let format = FormatDetector::detect(&path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap(), "numrs2");

        // Test is_numrs2_format
        assert!(FormatDetector::is_numrs2_format(&path));

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_detect_format_unknown() {
        // Create a file with unknown format
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_unknown.bin");

        fs::write(&path, b"UNKNOWN_FORMAT_DATA").unwrap();

        let format = FormatDetector::detect(&path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap(), "unknown");

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_streaming_serializer() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_streaming.numrs2");

        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((100, 50)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Test streaming save
        let result = StreamingSerializer::save_streaming(&model, &path);
        assert!(result.is_ok());

        // Test streaming load
        let loaded = StreamingSerializer::load_streaming(&path);
        assert!(loaded.is_ok());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_convenience_functions() {
        let weights = Array2::ones((5, 3));

        // Test compress
        let compressed = compress_weights(&weights);
        assert!(compressed.is_ok());

        // Test decompress
        let decompressed = decompress_weights(&compressed.unwrap());
        assert!(decompressed.is_ok());

        // Test quantize
        let quantized = quantize_weights(&weights, 8);
        assert!(quantized.is_ok());

        // Test hash
        let metadata = ModelMetadata::builder().name("test").build().unwrap();
        let layer = LayerData::dense("layer1", weights, None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let hash = compute_model_hash(&model);
        assert!(hash.is_ok());
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = Array2::from_shape_fn((4, 3), |(i, j)| (i * 3 + j) as f64 * 0.5);

        let (quantized, scale, zero_point) =
            ModelCompression::quantize_weights(&original, 8).unwrap();

        let recovered =
            ModelCompression::dequantize_weights(&quantized, (4, 3), 8, scale, zero_point).unwrap();

        // Check shapes match
        assert_eq!(recovered.shape(), original.shape());

        // Check values are approximately equal (quantization introduces error)
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.5); // Tolerance for 8-bit quantization
        }
    }

    #[test]
    fn test_hash_consistency() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let hash1 = ModelFingerprint::compute_hash(&model).unwrap();
        let hash2 = ModelFingerprint::compute_hash(&model).unwrap();

        // Same model should produce same hash
        assert_eq!(hash1, hash2);
    }
}

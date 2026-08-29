//! Model Export to Various Formats
//!
//! Provides export functionality for NumRS2 models to different formats
//! including JSON, MessagePack, and NPY/NPZ for interoperability.

use super::format::{FormatResult, LayerData, NumRS2Model};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Export format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON (human-readable)
    Json,
    /// MessagePack (compact binary)
    MessagePack,
    /// NumPy NPY format (single array)
    Npy,
    /// NumPy NPZ format (multiple arrays)
    Npz,
}

/// Model exporter for various formats
pub struct ModelExporter;

impl ModelExporter {
    /// Exports model to JSON format
    ///
    /// Creates a human-readable JSON representation of the model including
    /// metadata, architecture, and layer configuration (but not weights).
    ///
    /// # Arguments
    ///
    /// * `model` - The model to export
    /// * `path` - Output file path
    pub fn export_json<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        let export_data = ModelExportData::from_model(model);

        let file = File::create(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create JSON file: {}", e)))?;

        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &export_data).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize to JSON: {}", e))
        })?;

        Ok(())
    }

    /// Exports model to MessagePack format
    ///
    /// Creates a compact binary representation using MessagePack.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to export
    /// * `path` - Output file path
    #[cfg(feature = "messagepack")]
    pub fn export_messagepack<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        let export_data = ModelExportData::from_model(model);

        let file = File::create(path.as_ref()).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to create MessagePack file: {}", e))
        })?;

        let mut writer = BufWriter::new(file);

        let bytes = rmp_serde::to_vec(&export_data).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize to MessagePack: {}", e))
        })?;

        writer.write_all(&bytes).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to write MessagePack data: {}", e))
        })?;

        Ok(())
    }

    /// Exports model to MessagePack format (stub when feature is not enabled)
    #[cfg(not(feature = "messagepack"))]
    pub fn export_messagepack<P: AsRef<Path>>(_model: &NumRS2Model, _path: P) -> FormatResult<()> {
        Err(NumRs2Error::FeatureNotEnabled(
            "MessagePack export requires 'messagepack' feature".to_string(),
        ))
    }

    /// Exports model weights to NPZ format
    ///
    /// Saves all layer weights and biases as separate arrays in a compressed NPZ file.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to export
    /// * `path` - Output file path
    pub fn export_weights_npz<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        use oxiarc_archive::zip::ZipWriter;

        let file = File::create(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create NPZ file: {}", e)))?;

        let mut zip = ZipWriter::new(file);

        // Export each layer's weights
        for (i, layer) in model.layers.iter().enumerate() {
            // Export weights
            let weights_name = format!("layer_{}_weights.npy", i);
            Self::write_npy_to_zip(&mut zip, &weights_name, &layer.weights)?;

            // Export bias if present
            if let Some(ref bias) = layer.bias {
                let bias_name = format!("layer_{}_bias.npy", i);
                Self::write_npy_to_zip(&mut zip, &bias_name, bias)?;
            }
        }

        zip.finish()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to finish NPZ file: {}", e)))?;

        Ok(())
    }

    /// Helper function to write NPY data to ZIP
    fn write_npy_to_zip(
        zip: &mut oxiarc_archive::zip::ZipWriter<File>,
        name: &str,
        data: &[u8],
    ) -> FormatResult<()> {
        zip.add_file(name, data)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to add file to ZIP: {}", e)))?;

        Ok(())
    }

    /// Exports a single layer's weights to NPY format
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight array to export
    /// * `path` - Output file path
    pub fn export_weights_npy<P: AsRef<Path>>(weights: &Array2<f64>, path: P) -> FormatResult<()> {
        let file = File::create(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create NPY file: {}", e)))?;

        let mut writer = BufWriter::new(file);

        // Create NPY header
        let header = Self::create_npy_header(weights.shape(), "f8")?;
        writer
            .write_all(&header)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write NPY header: {}", e)))?;

        // Write data in C-order (row-major)
        use byteorder::{LittleEndian, WriteBytesExt};
        for &value in weights.iter() {
            writer
                .write_f64::<LittleEndian>(value)
                .map_err(|e| NumRs2Error::IOError(format!("Failed to write NPY data: {}", e)))?;
        }

        Ok(())
    }

    /// Creates NPY header
    fn create_npy_header(shape: &[usize], dtype: &str) -> FormatResult<Vec<u8>> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Magic string
        let magic = b"\x93NUMPY";

        // Version 1.0
        let version: [u8; 2] = [1, 0];

        // Create dictionary string
        let mut dict = format!(
            "{{'descr': '<{}', 'fortran_order': False, 'shape': (",
            dtype
        );

        for (i, &dim) in shape.iter().enumerate() {
            if i > 0 {
                dict.push_str(", ");
            }
            dict.push_str(&dim.to_string());

            // Add trailing comma for 1D arrays
            if shape.len() == 1 && i == shape.len() - 1 {
                dict.push(',');
            }
        }

        dict.push_str("), }");

        // Pad to make total header length a multiple of 16
        let header_len = 10 + dict.len(); // 6 (magic) + 2 (version) + 2 (len) + dict
        let padding = (16 - (header_len % 16)) % 16;
        dict.push_str(&" ".repeat(padding));

        // Build header
        let mut header = Vec::new();
        header.extend_from_slice(magic);
        header.extend_from_slice(&version);

        // Write header length (little endian)
        let dict_len = dict.len() as u16;
        header.write_u16::<LittleEndian>(dict_len).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to write header length: {}", e))
        })?;

        header.extend_from_slice(dict.as_bytes());

        Ok(header)
    }

    /// Exports model architecture description only
    ///
    /// Creates a JSON file with model architecture information without weights.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to export
    /// * `path` - Output file path
    pub fn export_architecture<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        let arch = ArchitectureDescription::from_model(model);

        let file = File::create(path.as_ref()).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to create architecture file: {}", e))
        })?;

        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &arch).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize architecture: {}", e))
        })?;

        Ok(())
    }
}

/// Model export data (without weight values)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExportData {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Architecture type
    pub architecture: String,
    /// Description
    pub description: Option<String>,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Layer information
    pub layers: Vec<LayerExportInfo>,
    /// Total parameters
    pub total_parameters: usize,
    /// Created timestamp
    pub created_at: String,
}

impl ModelExportData {
    /// Creates export data from a model
    pub fn from_model(model: &NumRS2Model) -> Self {
        let layers = model
            .layers
            .iter()
            .map(LayerExportInfo::from_layer)
            .collect();

        Self {
            name: model.metadata.name.clone(),
            version: model.metadata.version.clone(),
            architecture: model.metadata.architecture.clone(),
            description: model.metadata.description.clone(),
            hyperparameters: model.metadata.hyperparameters.clone(),
            layers,
            total_parameters: model.num_parameters(),
            created_at: model.metadata.created_at.clone(),
        }
    }
}

/// Layer export information (without weights)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerExportInfo {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub num_parameters: usize,
    /// Activation function
    pub activation: Option<String>,
    /// Layer parameters
    pub parameters: HashMap<String, String>,
}

impl LayerExportInfo {
    /// Creates layer export info from layer data
    pub fn from_layer(layer: &LayerData) -> Self {
        Self {
            name: layer.name.clone(),
            layer_type: format!("{:?}", layer.layer_type),
            input_shape: layer.input_shape.clone(),
            output_shape: layer.output_shape.clone(),
            num_parameters: layer.num_parameters(),
            activation: layer.activation.map(|a| format!("{:?}", a)),
            parameters: layer.parameters.clone(),
        }
    }
}

/// Architecture description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDescription {
    /// Architecture name
    pub name: String,
    /// Layer descriptions
    pub layers: Vec<LayerDescription>,
    /// Total parameters
    pub total_parameters: usize,
}

impl ArchitectureDescription {
    /// Creates architecture description from model
    pub fn from_model(model: &NumRS2Model) -> Self {
        let layers = model
            .layers
            .iter()
            .map(LayerDescription::from_layer)
            .collect();

        Self {
            name: model.metadata.architecture.clone(),
            layers,
            total_parameters: model.num_parameters(),
        }
    }
}

/// Layer description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDescription {
    /// Layer type
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Parameters
    pub parameters: HashMap<String, String>,
}

impl LayerDescription {
    /// Creates layer description from layer data
    pub fn from_layer(layer: &LayerData) -> Self {
        Self {
            layer_type: format!("{:?}", layer.layer_type),
            input_shape: layer.input_shape.clone(),
            output_shape: layer.output_shape.clone(),
            parameters: layer.parameters.clone(),
        }
    }
}

/// Convenience function to export model to JSON
pub fn export_to_json<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
    ModelExporter::export_json(model, path)
}

/// Convenience function to export model to MessagePack
pub fn export_to_messagepack<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
    ModelExporter::export_messagepack(model, path)
}

/// Convenience function to export weights to NPZ
pub fn export_weights_npz<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
    ModelExporter::export_weights_npz(model, path)
}

/// Convenience function to export weights to NPY
pub fn export_weights_npy<P: AsRef<Path>>(weights: &Array2<f64>, path: P) -> FormatResult<()> {
    ModelExporter::export_weights_npy(weights, path)
}

/// Convenience function to export architecture
pub fn export_architecture<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
    ModelExporter::export_architecture(model, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::model_io::format::{LayerData, ModelMetadata};
    use scirs2_core::ndarray::{Array1, Array2};
    use std::env;
    use std::fs;

    #[test]
    fn test_export_json() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_export.json");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .description("Test model for export")
            .hyperparameter("hidden_size", "128")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let result = ModelExporter::export_json(&model, &path);
        assert!(result.is_ok());

        // Verify file exists and is valid JSON
        assert!(path.exists());
        let contents = fs::read_to_string(&path).expect("test: valid file read");
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).expect("test: valid JSON parse");
        assert_eq!(parsed["name"], "test_model");
        assert_eq!(parsed["architecture"], "MLP");

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_export_architecture() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_architecture.json");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .architecture("Transformer")
            .build()
            .expect("test: valid metadata build");

        let layer1 = LayerData::dense("layer1", Array2::ones((512, 256)), None);
        let layer2 = LayerData::dense("layer2", Array2::ones((256, 128)), None);
        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        let result = ModelExporter::export_architecture(&model, &path);
        assert!(result.is_ok());

        // Verify file exists
        assert!(path.exists());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_export_weights_npy() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_weights.npy");

        let weights = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f64);

        let result = ModelExporter::export_weights_npy(&weights, &path);
        assert!(result.is_ok());

        // Verify file exists
        assert!(path.exists());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_npy_header_creation() {
        let shape = vec![3, 4];
        let header = ModelExporter::create_npy_header(&shape, "f8");
        assert!(header.is_ok());

        let header = header.expect("test: valid NPY header creation");
        assert!(header.starts_with(b"\x93NUMPY"));
        assert!(header.len().is_multiple_of(16)); // Should be aligned to 16 bytes
    }

    #[test]
    fn test_model_export_data_creation() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("CNN")
            .hyperparameter("kernel_size", "3")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let export_data = ModelExportData::from_model(&model);

        assert_eq!(export_data.name, "test_model");
        assert_eq!(export_data.version, "1.0.0");
        assert_eq!(export_data.architecture, "CNN");
        assert_eq!(export_data.layers.len(), 1);
        assert!(export_data.total_parameters > 0);
    }

    #[test]
    fn test_layer_export_info() {
        let weights = Array2::ones((10, 5));
        let layer = LayerData::dense("test_layer", weights, None);

        let info = LayerExportInfo::from_layer(&layer);

        assert_eq!(info.name, "test_layer");
        assert_eq!(info.layer_type, "Dense");
        assert_eq!(info.input_shape, vec![10]);
        assert_eq!(info.output_shape, vec![5]);
        assert!(info.num_parameters > 0);
    }

    #[test]
    fn test_architecture_description() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .architecture("ResNet")
            .build()
            .expect("test: valid metadata build");

        let layer1 = LayerData::dense("layer1", Array2::ones((256, 128)), None);
        let layer2 = LayerData::dense("layer2", Array2::ones((128, 64)), None);
        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        let arch = ArchitectureDescription::from_model(&model);

        assert_eq!(arch.name, "ResNet");
        assert_eq!(arch.layers.len(), 2);
        assert!(arch.total_parameters > 0);
    }

    #[test]
    fn test_export_format_enum() {
        assert_ne!(ExportFormat::Json, ExportFormat::MessagePack);
        assert_ne!(ExportFormat::Npy, ExportFormat::Npz);
    }

    #[test]
    fn test_convenience_functions() {
        let temp_dir = env::temp_dir();
        let json_path = temp_dir.join("test_convenience.json");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Test JSON export
        let result = export_to_json(&model, &json_path);
        assert!(result.is_ok());

        // Test architecture export
        let arch_path = temp_dir.join("test_arch.json");
        let result = export_architecture(&model, &arch_path);
        assert!(result.is_ok());

        // Cleanup
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(arch_path);
    }

    #[test]
    fn test_export_weights_npz() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_weights.npz");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer1 = LayerData::dense("layer1", Array2::ones((10, 5)), Some(Array1::zeros(5)));
        let layer2 = LayerData::dense("layer2", Array2::ones((5, 2)), None);
        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        let result = ModelExporter::export_weights_npz(&model, &path);
        assert!(result.is_ok());

        // Verify file exists
        assert!(path.exists());

        // Cleanup
        let _ = fs::remove_file(path);
    }
}

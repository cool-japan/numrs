//! Model Serialization and Checkpoint Management
//!
//! Provides comprehensive save/load functionality for NumRS2 models,
//! including checkpoint management, versioning, and incremental saves.

use super::format::{FormatResult, NumRS2Model};
use crate::error::NumRs2Error;
use serde_json;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Model serializer for save/load operations
pub struct ModelSerializer;

impl ModelSerializer {
    /// Saves a model to a file using Oxicode serialization
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `path` - Output file path
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use numrs2::new_modules::model_io::*;
    ///
    /// let model = NumRS2Model::new(metadata, layers);
    /// ModelSerializer::save(&model, "model.numrs2")?;
    /// ```
    pub fn save<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
        let file = File::create(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

        let mut writer = BufWriter::new(file);

        // Write NumRS2 magic bytes
        writer
            .write_all(b"NUMRS2\x00\x00")
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write magic bytes: {}", e)))?;

        // Serialize using serde_json
        let bytes = serde_json::to_vec(model).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize model: {}", e))
        })?;

        writer
            .write_all(&bytes)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write model data: {}", e)))?;

        writer
            .flush()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to flush writer: {}", e)))?;

        Ok(())
    }

    /// Loads a model from a file using Oxicode deserialization
    ///
    /// # Arguments
    ///
    /// * `path` - Input file path
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use numrs2::new_modules::model_io::*;
    ///
    /// let model = ModelSerializer::load("model.numrs2")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> FormatResult<NumRS2Model> {
        let file = File::open(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read magic bytes: {}", e)))?;

        if magic != *b"NUMRS2\x00\x00" {
            return Err(NumRs2Error::DeserializationError(
                "Invalid NumRS2 file format".to_string(),
            ));
        }

        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to read model data: {}", e)))?;

        // Deserialize using serde_json
        let model: NumRS2Model = serde_json::from_slice(&bytes).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to deserialize model: {}", e))
        })?;

        Ok(model)
    }

    /// Saves a model with compression
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `path` - Output file path
    ///
    /// # Returns
    ///
    /// Returns the compressed file size in bytes
    pub fn save_compressed<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<usize> {
        // First serialize to bytes
        let bytes = serde_json::to_vec(model).map_err(|e| {
            NumRs2Error::SerializationError(format!("Failed to serialize model: {}", e))
        })?;

        // Create ZIP archive with compression
        use oxiarc_archive::zip::{ZipCompressionLevel, ZipWriter};

        let file = File::create(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to create file: {}", e)))?;

        let mut zip = ZipWriter::new(file);

        // Add model data to archive
        zip.add_file("model.bin", &bytes)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to add file to ZIP: {}", e)))?;

        zip.finish()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to finish ZIP: {}", e)))?;

        let writer = zip
            .into_inner()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to get inner writer: {}", e)))?;

        // Get compressed size from file metadata
        let metadata = writer
            .metadata()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to get file metadata: {}", e)))?;

        Ok(metadata.len() as usize)
    }

    /// Loads a compressed model
    ///
    /// # Arguments
    ///
    /// * `path` - Input file path
    pub fn load_compressed<P: AsRef<Path>>(path: P) -> FormatResult<NumRS2Model> {
        use oxiarc_archive::zip::ZipReader;

        let file = File::open(path.as_ref())
            .map_err(|e| NumRs2Error::IOError(format!("Failed to open file: {}", e)))?;

        let mut zip = ZipReader::new(file)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to open ZIP: {}", e)))?;

        // Find and extract model.bin
        let entry = {
            let entries = zip.entries();
            entries
                .iter()
                .find(|e| e.name == "model.bin")
                .cloned()
                .ok_or_else(|| {
                    NumRs2Error::IOError("model.bin not found in ZIP archive".to_string())
                })?
        };

        let bytes = zip
            .extract(&entry)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to extract from ZIP: {}", e)))?;

        // Deserialize
        let model: NumRS2Model = serde_json::from_slice(&bytes).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to deserialize model: {}", e))
        })?;

        Ok(model)
    }
}

/// Checkpoint manager for tracking model versions and best models
pub struct CheckpointManager {
    /// Directory for storing checkpoints
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Track best model loss
    best_loss: Option<f64>,
    /// Best checkpoint path
    best_checkpoint: Option<PathBuf>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory for storing checkpoints
    /// * `max_checkpoints` - Maximum number of checkpoints to keep (0 = unlimited)
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, max_checkpoints: usize) -> FormatResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create checkpoint directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).map_err(|e| {
                NumRs2Error::IOError(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        Ok(Self {
            checkpoint_dir,
            max_checkpoints,
            best_loss: None,
            best_checkpoint: None,
        })
    }

    /// Saves a checkpoint
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `epoch` - Current epoch number
    /// * `loss` - Current loss value
    ///
    /// # Returns
    ///
    /// Returns the path to the saved checkpoint
    pub fn save_checkpoint(
        &mut self,
        model: &NumRS2Model,
        epoch: usize,
        loss: f64,
    ) -> FormatResult<PathBuf> {
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint_epoch_{:04}.numrs2", epoch));

        // Save checkpoint
        ModelSerializer::save(model, &checkpoint_path)?;

        // Update best model if necessary
        let is_best = self.best_loss.is_none_or(|best| loss < best);
        if is_best {
            self.best_loss = Some(loss);
            self.best_checkpoint = Some(checkpoint_path.clone());

            // Save as best model
            let best_path = self.checkpoint_dir.join("best_model.numrs2");
            ModelSerializer::save(model, &best_path)?;
        }

        // Clean up old checkpoints if necessary
        if self.max_checkpoints > 0 {
            self.cleanup_old_checkpoints()?;
        }

        Ok(checkpoint_path)
    }

    /// Loads the latest checkpoint
    pub fn load_latest(&self) -> FormatResult<NumRS2Model> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Err(NumRs2Error::IOError("No checkpoints found".to_string()));
        }

        // Get the latest checkpoint (last in sorted list)
        let latest = checkpoints
            .last()
            .ok_or_else(|| NumRs2Error::IOError("Failed to get latest checkpoint".to_string()))?;

        ModelSerializer::load(latest)
    }

    /// Loads the best checkpoint
    pub fn load_best(&self) -> FormatResult<NumRS2Model> {
        let best_path = self.checkpoint_dir.join("best_model.numrs2");

        if !best_path.exists() {
            return Err(NumRs2Error::IOError("No best model found".to_string()));
        }

        ModelSerializer::load(&best_path)
    }

    /// Lists all checkpoints
    pub fn list_checkpoints(&self) -> FormatResult<Vec<PathBuf>> {
        let entries = fs::read_dir(&self.checkpoint_dir).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to read checkpoint directory: {}", e))
        })?;

        let mut checkpoints: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("numrs2"))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("checkpoint_epoch_"))
            })
            .collect();

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Cleans up old checkpoints, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> FormatResult<()> {
        let mut checkpoints = self.list_checkpoints()?;

        // Keep only max_checkpoints most recent
        if checkpoints.len() > self.max_checkpoints {
            checkpoints.sort();
            let to_remove = checkpoints.len() - self.max_checkpoints;

            for checkpoint in checkpoints.iter().take(to_remove) {
                fs::remove_file(checkpoint).map_err(|e| {
                    NumRs2Error::IOError(format!("Failed to remove old checkpoint: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Gets the best loss value
    pub fn get_best_loss(&self) -> Option<f64> {
        self.best_loss
    }

    /// Gets the best checkpoint path
    pub fn get_best_checkpoint(&self) -> Option<&PathBuf> {
        self.best_checkpoint.as_ref()
    }
}

/// Model checkpoint with metadata
pub struct ModelCheckpoint {
    /// The model
    pub model: NumRS2Model,
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: Option<f64>,
    /// Timestamp
    pub timestamp: String,
}

impl ModelCheckpoint {
    /// Creates a new checkpoint
    pub fn new(model: NumRS2Model, epoch: usize, train_loss: f64, val_loss: Option<f64>) -> Self {
        Self {
            model,
            epoch,
            train_loss,
            val_loss,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Convenience function to save a model
pub fn save_model<P: AsRef<Path>>(model: &NumRS2Model, path: P) -> FormatResult<()> {
    ModelSerializer::save(model, path)
}

/// Convenience function to load a model
pub fn load_model<P: AsRef<Path>>(path: P) -> FormatResult<NumRS2Model> {
    ModelSerializer::load(path)
}

/// Convenience function to save a checkpoint
pub fn save_checkpoint<P: AsRef<Path>>(checkpoint: &ModelCheckpoint, path: P) -> FormatResult<()> {
    ModelSerializer::save(&checkpoint.model, path)
}

/// Convenience function to load a checkpoint
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> FormatResult<NumRS2Model> {
    ModelSerializer::load(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::model_io::format::{LayerData, ModelMetadata};
    use scirs2_core::ndarray::Array2;
    use std::env;

    #[test]
    fn test_save_and_load_model() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_model.numrs2");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Save model
        let result = ModelSerializer::save(&model, &path);
        assert!(result.is_ok());

        // Load model
        let loaded = ModelSerializer::load(&path);
        assert!(loaded.is_ok());

        let loaded_model = loaded.expect("test: valid model load");
        assert_eq!(loaded_model.metadata.name, "test_model");
        assert_eq!(loaded_model.num_layers(), 1);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_save_and_load_compressed() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_model_compressed.zip");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((100, 50)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Save compressed
        let result = ModelSerializer::save_compressed(&model, &path);
        assert!(result.is_ok());

        // Load compressed
        let loaded = ModelSerializer::load_compressed(&path);
        assert!(loaded.is_ok());

        let loaded_model = loaded.expect("test: valid compressed model load");
        assert_eq!(loaded_model.metadata.name, "test_model");

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_checkpoint_manager() {
        let temp_dir = env::temp_dir().join("test_checkpoints");
        let _ = fs::create_dir_all(&temp_dir);

        let mut manager =
            CheckpointManager::new(&temp_dir, 3).expect("test: valid checkpoint manager");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Save multiple checkpoints
        for epoch in 0..5 {
            let loss = 1.0 / (epoch + 1) as f64;
            let result = manager.save_checkpoint(&model, epoch, loss);
            assert!(result.is_ok());
        }

        // Check that only 3 checkpoints remain
        let checkpoints = manager
            .list_checkpoints()
            .expect("test: valid checkpoint listing");
        assert!(checkpoints.len() <= 3);

        // Load best model
        let best = manager.load_best();
        assert!(best.is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_checkpoint_best_tracking() {
        let temp_dir = env::temp_dir().join("test_best_tracking");
        let _ = fs::create_dir_all(&temp_dir);

        let mut manager =
            CheckpointManager::new(&temp_dir, 5).expect("test: valid checkpoint manager");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Save checkpoints with decreasing loss
        manager
            .save_checkpoint(&model, 0, 1.0)
            .expect("test: valid checkpoint save");
        manager
            .save_checkpoint(&model, 1, 0.5)
            .expect("test: valid checkpoint save");
        manager
            .save_checkpoint(&model, 2, 0.3)
            .expect("test: valid checkpoint save");
        manager
            .save_checkpoint(&model, 3, 0.7)
            .expect("test: valid checkpoint save"); // Not best

        // Best loss should be 0.3
        assert_eq!(manager.get_best_loss(), Some(0.3));

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_save_load_convenience_functions() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_convenience.numrs2");

        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        // Test convenience functions
        let result = save_model(&model, &path);
        assert!(result.is_ok());

        let loaded = load_model(&path);
        assert!(loaded.is_ok());

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_model("/nonexistent/path/model.numrs2");
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_manager_load_empty() {
        let temp_dir = env::temp_dir().join("test_empty_checkpoints");
        let _ = fs::create_dir_all(&temp_dir);

        let manager = CheckpointManager::new(&temp_dir, 3).expect("test: valid checkpoint manager");

        // Try to load from empty directory
        let result = manager.load_latest();
        assert!(result.is_err());

        let result = manager.load_best();
        assert!(result.is_err());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_model_checkpoint_creation() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let checkpoint = ModelCheckpoint::new(model, 10, 0.5, Some(0.6));

        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.train_loss, 0.5);
        assert_eq!(checkpoint.val_loss, Some(0.6));
        assert!(!checkpoint.timestamp.is_empty());
    }
}

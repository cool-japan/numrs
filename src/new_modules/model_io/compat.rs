//! Version Compatibility and Model Migration
//!
//! Provides utilities for migrating models between different NumRS2 versions,
//! validating model compatibility, and handling schema evolution.

use super::format::{FormatResult, ModelFormat, NumRS2Model, MODEL_FORMAT_VERSION};
use crate::error::NumRs2Error;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// Version information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
}

impl VersionInfo {
    /// Creates a new version info
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parses a version string (e.g., "0.4.0")
    pub fn parse(version: &str) -> FormatResult<Self> {
        let parts: Vec<&str> = version.split('.').collect();

        if parts.len() != 3 {
            return Err(NumRs2Error::ValueError(format!(
                "Invalid version string: {}",
                version
            )));
        }

        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| NumRs2Error::ValueError(format!("Invalid major version: {}", parts[0])))?;

        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| NumRs2Error::ValueError(format!("Invalid minor version: {}", parts[1])))?;

        let patch = parts[2]
            .parse::<u32>()
            .map_err(|_| NumRs2Error::ValueError(format!("Invalid patch version: {}", parts[2])))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Checks if this version is compatible with another
    ///
    /// Compatible if major version matches and this version is >= other
    pub fn is_compatible_with(&self, other: &VersionInfo) -> bool {
        self.major == other.major && self >= other
    }

    /// Checks if migration is needed
    pub fn needs_migration(&self, target: &VersionInfo) -> bool {
        self < target
    }
}

impl PartialOrd for VersionInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VersionInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => self.patch.cmp(&other.patch),
                other => other,
            },
            other => other,
        }
    }
}

impl fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Model validator
pub struct ModelValidator;

impl ModelValidator {
    /// Validates a model
    ///
    /// Checks:
    /// - Format version compatibility
    /// - Magic number correctness
    /// - Layer consistency
    /// - Metadata completeness
    pub fn validate(model: &NumRS2Model) -> FormatResult<()> {
        // Validate format
        Self::validate_format(&model.format)?;

        // Validate metadata
        Self::validate_metadata(model)?;

        // Validate layers
        Self::validate_layers(model)?;

        // Validate optimizer state if present
        if let Some(ref opt_state) = model.optimizer_state {
            Self::validate_optimizer_state(opt_state)?;
        }

        Ok(())
    }

    /// Validates model format
    fn validate_format(format: &ModelFormat) -> FormatResult<()> {
        // Check magic number
        if format.magic != *b"NUMRS2\x00\x00" {
            return Err(NumRs2Error::ValueError(
                "Invalid magic number in model format".to_string(),
            ));
        }

        // Check version
        let _version = VersionInfo::parse(&format.version)?;

        Ok(())
    }

    /// Validates model metadata
    fn validate_metadata(model: &NumRS2Model) -> FormatResult<()> {
        let metadata = &model.metadata;

        // Check required fields
        if metadata.name.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Model name cannot be empty".to_string(),
            ));
        }

        if metadata.version.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Model version cannot be empty".to_string(),
            ));
        }

        // Validate version format
        let _version = VersionInfo::parse(&metadata.version)?;

        Ok(())
    }

    /// Validates model layers
    fn validate_layers(model: &NumRS2Model) -> FormatResult<()> {
        if model.layers.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Model must have at least one layer".to_string(),
            ));
        }

        for (i, layer) in model.layers.iter().enumerate() {
            // Check layer name
            if layer.name.is_empty() {
                return Err(NumRs2Error::ValueError(format!(
                    "Layer {} has empty name",
                    i
                )));
            }

            // Check shapes
            if layer.input_shape.is_empty() {
                return Err(NumRs2Error::ValueError(format!(
                    "Layer {} has empty input shape",
                    layer.name
                )));
            }

            if layer.output_shape.is_empty() {
                return Err(NumRs2Error::ValueError(format!(
                    "Layer {} has empty output shape",
                    layer.name
                )));
            }

            // Check weights
            if layer.weights.is_empty() {
                return Err(NumRs2Error::ValueError(format!(
                    "Layer {} has no weights",
                    layer.name
                )));
            }
        }

        Ok(())
    }

    /// Validates optimizer state
    fn validate_optimizer_state(opt_state: &super::format::OptimizerState) -> FormatResult<()> {
        if opt_state.optimizer_name.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Optimizer name cannot be empty".to_string(),
            ));
        }

        if opt_state.learning_rate <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "Learning rate must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Model migration utilities
pub struct ModelMigration;

impl ModelMigration {
    /// Migrates a model to the current version
    ///
    /// # Arguments
    ///
    /// * `model` - The model to migrate
    ///
    /// # Returns
    ///
    /// A migrated model compatible with the current version
    pub fn migrate_to_current(mut model: NumRS2Model) -> FormatResult<NumRS2Model> {
        let current_version = VersionInfo::parse(MODEL_FORMAT_VERSION)?;
        let model_version = VersionInfo::parse(&model.format.version)?;

        if model_version == current_version {
            // No migration needed
            return Ok(model);
        }

        if !model_version.needs_migration(&current_version) {
            // Model is newer than current version
            return Err(NumRs2Error::ValueError(format!(
                "Model version {} is newer than supported version {}",
                model_version, current_version
            )));
        }

        // Perform migration based on version differences
        model = Self::migrate_from_version(model, &model_version, &current_version)?;

        // Update version
        model.format.version = MODEL_FORMAT_VERSION.to_string();

        Ok(model)
    }

    /// Migrates from a specific version to another
    fn migrate_from_version(
        mut model: NumRS2Model,
        from: &VersionInfo,
        to: &VersionInfo,
    ) -> FormatResult<NumRS2Model> {
        // Migration path: 0.1.x -> 0.2.x -> 0.3.x -> 0.4.x

        if from.major == 0 && from.minor < 2 {
            // Migrate from 0.1.x to 0.2.x
            model = Self::migrate_0_1_to_0_2(model)?;
        }

        if from.major == 0 && from.minor < 3 {
            // Migrate from 0.2.x to 0.3.x
            model = Self::migrate_0_2_to_0_3(model)?;
        }

        if from.major == 0 && from.minor < 4 {
            // Migrate from 0.3.x to 0.4.x
            model = Self::migrate_0_3_to_0_4(model)?;
        }

        Ok(model)
    }

    /// Migrates from 0.1.x to 0.2.x
    fn migrate_0_1_to_0_2(mut model: NumRS2Model) -> FormatResult<NumRS2Model> {
        // Add any 0.1.x -> 0.2.x specific migrations
        // For now, just update metadata
        model.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        Ok(model)
    }

    /// Migrates from 0.2.x to 0.3.x
    fn migrate_0_2_to_0_3(mut model: NumRS2Model) -> FormatResult<NumRS2Model> {
        // Add any 0.2.x -> 0.3.x specific migrations
        model.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        Ok(model)
    }

    /// Migrates from 0.3.x to 0.4.x
    fn migrate_0_3_to_0_4(mut model: NumRS2Model) -> FormatResult<NumRS2Model> {
        // Add any 0.3.x -> 0.4.x specific migrations
        model.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        Ok(model)
    }

    /// Checks if a model can be migrated
    pub fn can_migrate(model: &NumRS2Model) -> bool {
        let current_version = match VersionInfo::parse(MODEL_FORMAT_VERSION) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let model_version = match VersionInfo::parse(&model.format.version) {
            Ok(v) => v,
            Err(_) => return false,
        };

        // Can migrate if major version matches and model version is older
        current_version.major == model_version.major && model_version < current_version
    }
}

/// Convenience function to validate a model
pub fn validate_model_version(model: &NumRS2Model) -> FormatResult<()> {
    ModelValidator::validate(model)
}

/// Convenience function to migrate a model
pub fn migrate_model(model: NumRS2Model) -> FormatResult<NumRS2Model> {
    ModelMigration::migrate_to_current(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::model_io::format::{LayerData, ModelMetadata};
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_version_info_parsing() {
        let version = VersionInfo::parse("0.4.0");
        assert!(version.is_ok());

        let v = version.expect("test: valid version parse");
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 4);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_version_info_invalid() {
        let version = VersionInfo::parse("invalid");
        assert!(version.is_err());

        let version = VersionInfo::parse("1.2");
        assert!(version.is_err());

        let version = VersionInfo::parse("1.2.a");
        assert!(version.is_err());
    }

    #[test]
    fn test_version_comparison() {
        let v1 = VersionInfo::new(0, 4, 0);
        let v2 = VersionInfo::new(0, 3, 0);
        let v3 = VersionInfo::new(0, 4, 1);

        assert!(v1 > v2);
        assert!(v3 > v1);
        assert!(v2 < v1);
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = VersionInfo::new(0, 4, 0);
        let v2 = VersionInfo::new(0, 3, 0);
        let v3 = VersionInfo::new(1, 0, 0);

        assert!(v1.is_compatible_with(&v2)); // Same major, v1 >= v2
        assert!(!v2.is_compatible_with(&v1)); // Same major, but v2 < v1
        assert!(!v1.is_compatible_with(&v3)); // Different major
    }

    #[test]
    fn test_version_needs_migration() {
        let v1 = VersionInfo::new(0, 3, 0);
        let v2 = VersionInfo::new(0, 4, 0);

        assert!(v1.needs_migration(&v2));
        assert!(!v2.needs_migration(&v1));
    }

    #[test]
    fn test_version_to_string() {
        let v = VersionInfo::new(0, 4, 0);
        assert_eq!(v.to_string(), "0.4.0");
    }

    #[test]
    fn test_validate_model_success() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let model = NumRS2Model::new(metadata, vec![layer]);

        let result = ModelValidator::validate(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_empty_name() {
        let mut model = create_test_model();
        model.metadata.name = String::new();

        let result = ModelValidator::validate(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_no_layers() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .build()
            .expect("test: valid metadata build");

        let model = NumRS2Model::new(metadata, vec![]);

        let result = ModelValidator::validate(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_format() {
        let format = ModelFormat::default();
        let result = ModelValidator::validate_format(&format);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_format_invalid_magic() {
        let format = ModelFormat {
            magic: *b"INVALID\x00",
            ..Default::default()
        };

        let result = ModelValidator::validate_format(&format);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_migrate() {
        let mut model = create_test_model();
        model.format.version = "0.3.0".to_string();

        let can_migrate = ModelMigration::can_migrate(&model);
        assert!(can_migrate);
    }

    #[test]
    fn test_can_migrate_same_version() {
        let model = create_test_model();
        let can_migrate = ModelMigration::can_migrate(&model);
        // Should be false since it's already at current version
        assert!(!can_migrate);
    }

    #[test]
    fn test_migrate_to_current() {
        let mut model = create_test_model();
        model.format.version = "0.3.0".to_string();

        let result = ModelMigration::migrate_to_current(model);
        assert!(result.is_ok());

        let migrated = result.expect("test: valid model migration");
        assert_eq!(migrated.format.version, MODEL_FORMAT_VERSION);
    }

    #[test]
    fn test_migrate_newer_version() {
        let mut model = create_test_model();
        model.format.version = "1.0.0".to_string(); // Future version

        let result = ModelMigration::migrate_to_current(model);
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_functions() {
        let model = create_test_model();

        // Test validation
        let result = validate_model_version(&model);
        assert!(result.is_ok());

        // Test migration (should work even with current version)
        let result = migrate_model(model);
        assert!(result.is_ok());
    }

    // Helper function
    fn create_test_model() -> NumRS2Model {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .build()
            .expect("test: valid metadata build");

        let layer = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        NumRS2Model::new(metadata, vec![layer])
    }

    #[test]
    fn test_validate_optimizer_state() {
        let opt_state = super::super::format::OptimizerState::adam(0.001, 0.9, 0.999, 1e-8);
        let result = ModelValidator::validate_optimizer_state(&opt_state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_optimizer_state_invalid_lr() {
        let mut opt_state = super::super::format::OptimizerState::adam(0.001, 0.9, 0.999, 1e-8);
        opt_state.learning_rate = -0.1;

        let result = ModelValidator::validate_optimizer_state(&opt_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_version_ordering() {
        let mut versions = [
            VersionInfo::new(0, 4, 0),
            VersionInfo::new(0, 2, 0),
            VersionInfo::new(0, 3, 5),
            VersionInfo::new(0, 3, 0),
        ];

        versions.sort();

        assert_eq!(versions[0], VersionInfo::new(0, 2, 0));
        assert_eq!(versions[1], VersionInfo::new(0, 3, 0));
        assert_eq!(versions[2], VersionInfo::new(0, 3, 5));
        assert_eq!(versions[3], VersionInfo::new(0, 4, 0));
    }
}

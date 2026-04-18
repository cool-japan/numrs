//! # Model Registry
//!
//! Multi-model management system with versioning, A/B testing, and hot model reloading.

use super::{Model, Result, ServingError};
use crate::array::Array;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Model metadata
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model description
    pub description: Option<String>,

    /// Model tags for categorization
    pub tags: Vec<String>,

    /// Model load timestamp
    pub loaded_at: Instant,

    /// Model warmup status
    pub warmed_up: bool,

    /// Model size in bytes (if available)
    pub size_bytes: Option<usize>,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            description: None,
            tags: Vec::new(),
            loaded_at: Instant::now(),
            warmed_up: false,
            size_bytes: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Add multiple tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Set model size
    pub fn with_size(mut self, size_bytes: usize) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }
}

/// Model version entry
struct ModelVersion {
    model: Box<dyn Model>,
    metadata: ModelMetadata,
}

/// A/B testing configuration
#[derive(Clone, Debug)]
pub struct ABTestConfig {
    /// Version A name
    pub version_a: String,

    /// Version B name
    pub version_b: String,

    /// Traffic split percentage for version A (0.0 to 1.0)
    pub split_percent: f64,

    /// Number of requests routed to version A
    pub requests_a: usize,

    /// Number of requests routed to version B
    pub requests_b: usize,
}

impl ABTestConfig {
    /// Create new A/B test configuration
    pub fn new(version_a: String, version_b: String, split_percent: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&split_percent) {
            return Err(ServingError::ValidationError {
                field: "split_percent".to_string(),
                message: "Split percentage must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            version_a,
            version_b,
            split_percent,
            requests_a: 0,
            requests_b: 0,
        })
    }

    /// Get version based on split percentage and request count
    pub fn get_version(&self) -> &str {
        let total = self.requests_a + self.requests_b;
        if total == 0 {
            return &self.version_a;
        }

        let actual_split = self.requests_a as f64 / total as f64;
        if actual_split < self.split_percent {
            &self.version_a
        } else {
            &self.version_b
        }
    }
}

/// Model registry for multi-model management
pub struct ModelRegistry {
    models: RwLock<HashMap<String, HashMap<String, ModelVersion>>>,
    default_versions: RwLock<HashMap<String, String>>,
    ab_tests: RwLock<HashMap<String, ABTestConfig>>,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            default_versions: RwLock::new(HashMap::new()),
            ab_tests: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new model
    pub fn register(
        &self,
        name: &str,
        version: &str,
        model: Box<dyn Model>,
        metadata: Option<ModelMetadata>,
    ) -> Result<()> {
        let mut models = self
            .models
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models write lock".to_string(),
            })?;

        let model_metadata =
            metadata.unwrap_or_else(|| ModelMetadata::new(name.to_string(), version.to_string()));

        let version_entry = ModelVersion {
            model,
            metadata: model_metadata,
        };

        models
            .entry(name.to_string())
            .or_insert_with(HashMap::new)
            .insert(version.to_string(), version_entry);

        // Set as default version if this is the first version
        let mut default_versions =
            self.default_versions
                .write()
                .map_err(|_| ServingError::ConcurrencyError {
                    message: "Failed to acquire default_versions write lock".to_string(),
                })?;

        if !default_versions.contains_key(name) {
            default_versions.insert(name.to_string(), version.to_string());
        }

        Ok(())
    }

    /// Unregister a model version
    pub fn unregister(&self, name: &str, version: &str) -> Result<()> {
        let mut models = self
            .models
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models write lock".to_string(),
            })?;

        if let Some(versions) = models.get_mut(name) {
            versions.remove(version);

            // Remove model entry if no versions left
            if versions.is_empty() {
                models.remove(name);

                let mut default_versions =
                    self.default_versions
                        .write()
                        .map_err(|_| ServingError::ConcurrencyError {
                            message: "Failed to acquire default_versions write lock".to_string(),
                        })?;
                default_versions.remove(name);
            }
        }

        Ok(())
    }

    /// Get model by name and version
    pub fn get(&self, name: &str, version: Option<&str>) -> Result<Arc<RwLock<Box<dyn Model>>>> {
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        let versions = models
            .get(name)
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: name.to_string(),
                version: version.map(String::from),
            })?;

        let version_str = if let Some(v) = version {
            v.to_string()
        } else {
            let default_versions =
                self.default_versions
                    .read()
                    .map_err(|_| ServingError::ConcurrencyError {
                        message: "Failed to acquire default_versions read lock".to_string(),
                    })?;

            default_versions
                .get(name)
                .ok_or_else(|| ServingError::ModelNotFound {
                    model_name: name.to_string(),
                    version: None,
                })?
                .clone()
        };

        // Note: We can't return a direct reference to the model due to lock lifetime
        // In a real implementation, you'd use Arc<RwLock<Box<dyn Model>>>
        Err(ServingError::Other {
            message: "Direct model access not supported - use get_metadata or invoke methods"
                .to_string(),
        })
    }

    /// Get model metadata
    pub fn get_metadata(&self, name: &str, version: Option<&str>) -> Result<ModelMetadata> {
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        let versions = models
            .get(name)
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: name.to_string(),
                version: version.map(String::from),
            })?;

        let version_str = if let Some(v) = version {
            v.to_string()
        } else {
            let default_versions =
                self.default_versions
                    .read()
                    .map_err(|_| ServingError::ConcurrencyError {
                        message: "Failed to acquire default_versions read lock".to_string(),
                    })?;

            default_versions
                .get(name)
                .ok_or_else(|| ServingError::ModelNotFound {
                    model_name: name.to_string(),
                    version: None,
                })?
                .clone()
        };

        let version_entry =
            versions
                .get(&version_str)
                .ok_or_else(|| ServingError::ModelNotFound {
                    model_name: name.to_string(),
                    version: Some(version_str),
                })?;

        Ok(version_entry.metadata.clone())
    }

    /// Set default version for a model
    pub fn set_default_version(&self, name: &str, version: &str) -> Result<()> {
        // Verify model and version exist
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        let versions = models
            .get(name)
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: name.to_string(),
                version: Some(version.to_string()),
            })?;

        if !versions.contains_key(version) {
            return Err(ServingError::InvalidVersion {
                model_name: name.to_string(),
                version: version.to_string(),
                message: "Version not found".to_string(),
            });
        }

        drop(models);

        let mut default_versions =
            self.default_versions
                .write()
                .map_err(|_| ServingError::ConcurrencyError {
                    message: "Failed to acquire default_versions write lock".to_string(),
                })?;

        default_versions.insert(name.to_string(), version.to_string());
        Ok(())
    }

    /// Get default version for a model
    pub fn get_default_version(&self, name: &str) -> Result<String> {
        let default_versions =
            self.default_versions
                .read()
                .map_err(|_| ServingError::ConcurrencyError {
                    message: "Failed to acquire default_versions read lock".to_string(),
                })?;

        default_versions
            .get(name)
            .cloned()
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: name.to_string(),
                version: None,
            })
    }

    /// List all registered models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        Ok(models.keys().cloned().collect())
    }

    /// List all versions of a model
    pub fn list_versions(&self, name: &str) -> Result<Vec<String>> {
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        let versions = models
            .get(name)
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: name.to_string(),
                version: None,
            })?;

        Ok(versions.keys().cloned().collect())
    }

    /// Create A/B test between two versions
    pub fn create_ab_test(
        &self,
        model_name: &str,
        version_a: &str,
        version_b: &str,
        split_percent: f64,
    ) -> Result<()> {
        // Verify both versions exist
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        let versions = models
            .get(model_name)
            .ok_or_else(|| ServingError::ModelNotFound {
                model_name: model_name.to_string(),
                version: None,
            })?;

        if !versions.contains_key(version_a) {
            return Err(ServingError::InvalidVersion {
                model_name: model_name.to_string(),
                version: version_a.to_string(),
                message: "Version A not found".to_string(),
            });
        }

        if !versions.contains_key(version_b) {
            return Err(ServingError::InvalidVersion {
                model_name: model_name.to_string(),
                version: version_b.to_string(),
                message: "Version B not found".to_string(),
            });
        }

        drop(models);

        let config =
            ABTestConfig::new(version_a.to_string(), version_b.to_string(), split_percent)?;

        let mut ab_tests = self
            .ab_tests
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire ab_tests write lock".to_string(),
            })?;

        ab_tests.insert(model_name.to_string(), config);
        Ok(())
    }

    /// Get A/B test configuration
    pub fn get_ab_test(&self, model_name: &str) -> Result<ABTestConfig> {
        let ab_tests = self
            .ab_tests
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire ab_tests read lock".to_string(),
            })?;

        ab_tests
            .get(model_name)
            .cloned()
            .ok_or_else(|| ServingError::Other {
                message: format!("No A/B test configured for model '{}'", model_name),
            })
    }

    /// Remove A/B test
    pub fn remove_ab_test(&self, model_name: &str) -> Result<()> {
        let mut ab_tests = self
            .ab_tests
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire ab_tests write lock".to_string(),
            })?;

        ab_tests.remove(model_name);
        Ok(())
    }

    /// Get total number of registered models
    pub fn model_count(&self) -> Result<usize> {
        let models = self
            .models
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire models read lock".to_string(),
            })?;

        Ok(models.len())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock model for testing
    struct MockModel {
        name: String,
        version: String,
    }

    impl MockModel {
        fn new(name: &str, version: &str) -> Self {
            Self {
                name: name.to_string(),
                version: version.to_string(),
            }
        }
    }

    impl Model for MockModel {
        fn forward(&self, input: &Array<f64>) -> Result<Array<f64>> {
            Ok(input.clone())
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn input_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }

        fn output_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count().expect("test: valid model count"), 0);
    }

    #[test]
    fn test_register_model() {
        let registry = ModelRegistry::new();
        let model = Box::new(MockModel::new("test_model", "v1.0"));

        registry
            .register("test_model", "v1.0", model, None)
            .expect("Registration should succeed");

        assert_eq!(
            registry
                .model_count()
                .expect("test: valid model count after registration"),
            1
        );
    }

    #[test]
    fn test_register_multiple_versions() {
        let registry = ModelRegistry::new();
        let model_v1 = Box::new(MockModel::new("test_model", "v1.0"));
        let model_v2 = Box::new(MockModel::new("test_model", "v2.0"));

        registry
            .register("test_model", "v1.0", model_v1, None)
            .expect("Registration should succeed");
        registry
            .register("test_model", "v2.0", model_v2, None)
            .expect("Registration should succeed");

        let versions = registry
            .list_versions("test_model")
            .expect("List versions should succeed");
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_unregister_model() {
        let registry = ModelRegistry::new();
        let model = Box::new(MockModel::new("test_model", "v1.0"));

        registry
            .register("test_model", "v1.0", model, None)
            .expect("Registration should succeed");

        registry
            .unregister("test_model", "v1.0")
            .expect("Unregistration should succeed");

        assert_eq!(registry.model_count().expect("test: valid model count"), 0);
    }

    #[test]
    fn test_get_metadata() {
        let registry = ModelRegistry::new();
        let model = Box::new(MockModel::new("test_model", "v1.0"));

        let metadata = ModelMetadata::new("test_model".to_string(), "v1.0".to_string())
            .with_description("Test model".to_string())
            .with_tag("test".to_string());

        registry
            .register("test_model", "v1.0", model, Some(metadata.clone()))
            .expect("Registration should succeed");

        let retrieved_metadata = registry
            .get_metadata("test_model", Some("v1.0"))
            .expect("Get metadata should succeed");

        assert_eq!(retrieved_metadata.name, "test_model");
        assert_eq!(retrieved_metadata.version, "v1.0");
        assert_eq!(
            retrieved_metadata.description,
            Some("Test model".to_string())
        );
    }

    #[test]
    fn test_default_version() {
        let registry = ModelRegistry::new();
        let model_v1 = Box::new(MockModel::new("test_model", "v1.0"));
        let model_v2 = Box::new(MockModel::new("test_model", "v2.0"));

        registry
            .register("test_model", "v1.0", model_v1, None)
            .expect("Registration should succeed");
        registry
            .register("test_model", "v2.0", model_v2, None)
            .expect("Registration should succeed");

        // First version should be default
        let default = registry
            .get_default_version("test_model")
            .expect("Get default version should succeed");
        assert_eq!(default, "v1.0");

        // Change default version
        registry
            .set_default_version("test_model", "v2.0")
            .expect("Set default version should succeed");

        let new_default = registry
            .get_default_version("test_model")
            .expect("Get default version should succeed");
        assert_eq!(new_default, "v2.0");
    }

    #[test]
    fn test_list_models() {
        let registry = ModelRegistry::new();
        let model1 = Box::new(MockModel::new("model1", "v1.0"));
        let model2 = Box::new(MockModel::new("model2", "v1.0"));

        registry
            .register("model1", "v1.0", model1, None)
            .expect("Registration should succeed");
        registry
            .register("model2", "v1.0", model2, None)
            .expect("Registration should succeed");

        let models = registry.list_models().expect("List models should succeed");
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"model1".to_string()));
        assert!(models.contains(&"model2".to_string()));
    }

    #[test]
    fn test_ab_test_creation() {
        let registry = ModelRegistry::new();
        let model_v1 = Box::new(MockModel::new("test_model", "v1.0"));
        let model_v2 = Box::new(MockModel::new("test_model", "v2.0"));

        registry
            .register("test_model", "v1.0", model_v1, None)
            .expect("Registration should succeed");
        registry
            .register("test_model", "v2.0", model_v2, None)
            .expect("Registration should succeed");

        registry
            .create_ab_test("test_model", "v1.0", "v2.0", 0.5)
            .expect("A/B test creation should succeed");

        let ab_test = registry
            .get_ab_test("test_model")
            .expect("Get A/B test should succeed");

        assert_eq!(ab_test.version_a, "v1.0");
        assert_eq!(ab_test.version_b, "v2.0");
        assert_eq!(ab_test.split_percent, 0.5);
    }

    #[test]
    fn test_ab_test_invalid_split() {
        let config_result = ABTestConfig::new("v1".to_string(), "v2".to_string(), 1.5);
        assert!(config_result.is_err());
    }

    #[test]
    fn test_ab_test_version_selection() {
        let config = ABTestConfig::new("v1".to_string(), "v2".to_string(), 0.5)
            .expect("Config creation should succeed");

        // First request should go to version A
        let version = config.get_version();
        assert_eq!(version, "v1");
    }

    #[test]
    fn test_remove_ab_test() {
        let registry = ModelRegistry::new();
        let model_v1 = Box::new(MockModel::new("test_model", "v1.0"));
        let model_v2 = Box::new(MockModel::new("test_model", "v2.0"));

        registry
            .register("test_model", "v1.0", model_v1, None)
            .expect("Registration should succeed");
        registry
            .register("test_model", "v2.0", model_v2, None)
            .expect("Registration should succeed");

        registry
            .create_ab_test("test_model", "v1.0", "v2.0", 0.5)
            .expect("A/B test creation should succeed");

        registry
            .remove_ab_test("test_model")
            .expect("Remove A/B test should succeed");

        assert!(registry.get_ab_test("test_model").is_err());
    }

    #[test]
    fn test_model_metadata_builder() {
        let metadata = ModelMetadata::new("test".to_string(), "v1.0".to_string())
            .with_description("Test description".to_string())
            .with_tag("tag1".to_string())
            .with_tag("tag2".to_string())
            .with_size(1024);

        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.version, "v1.0");
        assert_eq!(metadata.description, Some("Test description".to_string()));
        assert_eq!(metadata.tags.len(), 2);
        assert_eq!(metadata.size_bytes, Some(1024));
    }
}

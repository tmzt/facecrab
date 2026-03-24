use anyhow::{Context, Result};
use crate::core_types::{ModelFormat, ModelSpec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

const DEFAULT_MODELS: &str = include_str!("models.toml");

#[derive(Debug, Serialize, Deserialize)]
struct RegistryFile {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelPurpose {
    Inference,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub repo: String,
    pub filename: String,
    pub quantization: String,
    #[serde(default = "default_purpose")]
    pub purpose: ModelPurpose,
    /// Model format: "Gguf" (default) or "Safetensors".
    #[serde(default)]
    pub format: ModelFormat,
    /// For multi-file formats (Safetensors), the list of files to download.
    #[serde(default)]
    pub files: Vec<String>,
}

fn default_purpose() -> ModelPurpose {
    ModelPurpose::Inference
}

pub struct ModelRegistry {
    config_dir: PathBuf,
    cache_dir: PathBuf,
    models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let config_dir = if let Ok(home) = std::env::var("GENIUS_HOME") {
            PathBuf::from(home)
        } else if let Ok(custom_path) = std::env::var("RUSTY_GENIUS_CONFIG_DIR") {
            PathBuf::from(custom_path)
        } else {
            dirs::config_dir()
                .context("Could not find config directory")?
                .join("rusty-genius")
        };

        // Resolve Cache Directory
        let cache_dir = if let Ok(cache) = std::env::var("GENIUS_CACHE") {
            PathBuf::from(cache)
        } else {
            config_dir.join("cache")
        };

        fs::create_dir_all(&config_dir)?;
        fs::create_dir_all(&cache_dir)?;

        let mut registry = Self {
            config_dir,
            cache_dir,
            models: HashMap::new(),
        };

        registry.load_defaults()?;
        registry.load_manifest()?;
        registry.load_dynamic()?;

        Ok(registry)
    }

    fn load_defaults(&mut self) -> Result<()> {
        let parsed: RegistryFile = toml::from_str(DEFAULT_MODELS)?;
        for model in parsed.models {
            self.models.insert(model.name.clone(), model);
        }
        Ok(())
    }

    fn load_manifest(&mut self) -> Result<()> {
        let manifest_path = self.config_dir.join("manifest.toml");
        if manifest_path.exists() {
            let content = fs::read_to_string(manifest_path)?;
            let parsed: RegistryFile = toml::from_str(&content)?;
            for model in parsed.models {
                self.models.insert(model.name.clone(), model);
            }
        }
        Ok(())
    }

    fn load_dynamic(&mut self) -> Result<()> {
        let registry_path = self.cache_dir.join("registry.toml");
        if registry_path.exists() {
            let content = fs::read_to_string(registry_path)?;
            let parsed: RegistryFile = toml::from_str(&content)?;
            for model in parsed.models {
                self.models.insert(model.name.clone(), model);
            }
        }
        Ok(())
    }

    pub fn record_model(&mut self, entry: ModelEntry) -> Result<()> {
        // Add to in-memory map
        self.models.insert(entry.name.clone(), entry.clone());

        // Save to cache_dir/registry.toml
        let registry_path = self.cache_dir.join("registry.toml");
        let mut entries = Vec::new();

        // If it exists, read existing ones to preserve them
        if registry_path.exists() {
            let content = fs::read_to_string(&registry_path)?;
            if let Ok(parsed) = toml::from_str::<RegistryFile>(&content) {
                entries = parsed.models;
            }
        }

        // Add or update entry
        if let Some(pos) = entries.iter().position(|e| e.name == entry.name) {
            entries[pos] = entry;
        } else {
            entries.push(entry);
        }

        let new_content = toml::to_string(&RegistryFile { models: entries })?;
        fs::write(registry_path, new_content)?;

        Ok(())
    }

    pub fn list_models(&self) -> Vec<ModelEntry> {
        self.models.values().cloned().collect()
    }

    pub fn resolve(&self, name_or_spec: &str) -> Option<ModelSpec> {
        if let Some(entry) = self.models.get(name_or_spec) {
            return Some(ModelSpec {
                repo: entry.repo.clone(),
                filename: entry.filename.clone(),
                quantization: entry.quantization.clone(),
                format: entry.format.clone(),
                files: entry.files.clone(),
            });
        }
        None
    }

    pub fn get_cache_dir(&self) -> PathBuf {
        self.cache_dir.clone()
    }
}

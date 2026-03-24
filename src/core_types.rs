//! Core types inlined from rusty-genius-core.
//! These were previously in a separate crate; now self-contained.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Manifest types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    Mlx,
}

impl Default for ModelFormat {
    fn default() -> Self { ModelFormat::Gguf }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub repo: String,
    pub filename: String,
    pub quantization: String,
    #[serde(default)]
    pub format: ModelFormat,
    #[serde(default)]
    pub files: Vec<String>,
}

// ── Protocol types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetEvent {
    Started(String),
    Progress(u64, u64),
    Complete(String),
    Error(String),
}

// ── Error types ─────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum GeniusError {
    #[error("Protocol Error: {0}")]
    ProtocolError(String),
    #[error("Manifest Error: {0}")]
    ManifestError(String),
    #[error("Asset Error: {0}")]
    AssetError(String),
    #[error("Engine Error: {0}")]
    EngineError(String),
    #[error("Memory Error: {0}")]
    MemoryError(String),
    #[error("Unknown Error: {0}")]
    Unknown(String),
}

//! # Facecrab: The Supplier
//!
//! **Asset management, model registry, and high-performance LLM downloads.**
//!
//! Facecrab handles model resolution (via HuggingFace), registry management,
//! and background downloading with progress tracking.
//!
//! ## Core Features
//!
//! - **Registry Management**: Uses `registry.toml` to map friendly names to HuggingFace repositories.
//! - **HuggingFace Integration**: Automatically resolves and downloads GGUF assets.
//! - **Streaming Downloads**: Provides an event-based API for tracking download progress (bytes/total).
//! - **Local Caching**: Deduplicates downloads and manages assets in `~/.config/rusty-genius/`.
//!
//! ## Usage
//!
//! ### 1. Simple One-Shot Download
//!
//! ```no_run
//! use facecrab::AssetAuthority;
//!
//! #[async_std::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let authority = AssetAuthority::new()?;
//!     let path = authority.ensure_model("qwen-2.5-3b-instruct").await?;
//!     println!("Model available at: {:?}", path);
//!     Ok(())
//! }
//! ```
//!
//! ### 2. Event-Based Download (Progress Tracking)
//!
//! ```no_run
//! use facecrab::{AssetAuthority, AssetEvent};
//! use futures::StreamExt;
//!
//! #[async_std::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let authority = AssetAuthority::new()?;
//!     let mut events = authority.ensure_model_stream("tiny-model");
//!
//!     while let Some(event) = events.next().await {
//!         match event {
//!             AssetEvent::Started(name) => println!("Starting download: {}", name),
//!             AssetEvent::Progress(current, total) => {
//!                 let pct = (current as f64 / total as f64) * 100.0;
//!                 print!("\rProgress: {:.2}% ({}/{})", pct, current, total);
//!             }
//!             AssetEvent::Complete(path) => println!("\nModel ready at: {}", path),
//!             AssetEvent::Error(err) => eprintln!("Error: {}", err),
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

/// Core types (inlined from rusty-genius-core).
pub mod core_types;

/// Logic for downloading and caching assets from remote sources.
pub mod assets;

/// Management of the local model registry and configuration.
pub mod registry;

pub use assets::AssetAuthority;
pub use registry::ModelRegistry;
pub use core_types::{AssetEvent, GeniusError, ModelFormat, ModelSpec};

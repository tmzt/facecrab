use crate::registry::ModelEntry;
use crate::registry::ModelRegistry;
use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use crate::core_types::{ModelFormat, ModelSpec, AssetEvent, GeniusError};
use std::fs;
use std::path::PathBuf;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

struct ProgressReader<R> {
    inner: R,
    current: u64,
    total: u64,
    sender: mpsc::Sender<AssetEvent>,
}

impl<R: futures::io::AsyncRead + Unpin> futures::io::AsyncRead for ProgressReader<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match std::pin::Pin::new(&mut self.inner).poll_read(cx, buf) {
            std::task::Poll::Ready(Ok(n)) => {
                if n > 0 {
                    self.current += n as u64;
                    let current = self.current;
                    let total = self.total;
                    let _ = self.sender.try_send(AssetEvent::Progress(current, total));
                }
                std::task::Poll::Ready(Ok(n))
            }
            other => other,
        }
    }
}

impl AssetAuthority {
    pub fn new() -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::new()?,
        })
    }

    /// Access the underlying model registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// List all models in the registry.
    pub fn list_models(&self) -> Vec<ModelEntry> {
        self.registry.list_models()
    }

    /// Download a model and return its local path.
    pub async fn ensure_model(&self, name: &str) -> Result<PathBuf> {
        let (tx, mut rx) = mpsc::channel(1);
        let name = name.to_string();

        let handle = async_std::task::spawn(async move {
            if let Ok(auth) = AssetAuthority::new() {
                auth.ensure_model_internal(&name, tx, true).await
            } else {
                Err(anyhow::anyhow!("Failed to create authority"))
            }
        });

        while rx.next().await.is_some() {}
        handle.await
    }

    /// Download a model from a spec and return a stream of [AssetEvent]s.
    /// Use this for programmatic downloads where you already know the repo/files.
    pub fn ensure_spec_stream(&self, spec: ModelSpec) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);

        async_std::task::spawn(async move {
            let mut err_tx = tx.clone();
            let result: Result<()> = async {
                let auth = AssetAuthority::new()?;
                let name = spec.filename.clone();
                match spec.format {
                    ModelFormat::Safetensors => {
                        auth.ensure_safetensors_model(&name, &spec, tx, false).await?;
                    }
                    ModelFormat::Gguf => {
                        auth.ensure_gguf_model(&name, &spec, tx, false).await?;
                    }
                }
                Ok(())
            }
            .await;

            if let Err(e) = result {
                let _ = err_tx.send(AssetEvent::Error(e.to_string())).await;
            }
        });

        rx
    }

    /// Download a model and return a stream of [AssetEvent]s.
    pub fn ensure_model_stream(&self, name: &str) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);
        let name = name.to_string();

        async_std::task::spawn(async move {
            let mut err_tx = tx.clone();
            let result: Result<()> = async {
                let auth = AssetAuthority::new()?;
                auth.ensure_model_internal(&name, tx, false).await?;
                Ok(())
            }
            .await;

            if let Err(e) = result {
                let _ = err_tx.send(AssetEvent::Error(e.to_string())).await;
            }
        });

        rx
    }

    async fn ensure_model_internal(
        &self,
        name: &str,
        mut tx: mpsc::Sender<AssetEvent>,
        silent: bool,
    ) -> Result<PathBuf> {
        let _ = tx.send(AssetEvent::Started(name.to_string())).await;

        let spec = if let Some(s) = self.registry.resolve(name) {
            s
        } else if name.contains('/') {
            let parts: Vec<&str> = name.split(':').collect();
            if parts.len() >= 2 {
                ModelSpec {
                    repo: parts[0].to_string(),
                    filename: parts[1].to_string(),
                    quantization: parts.get(2).unwrap_or(&"Q4_K_M").to_string(),
                    format: ModelFormat::Gguf,
                    files: vec![],
                }
            } else {
                let err = format!(
                    "Model '{}' not found and invalid Repo/Repo:filename format",
                    name
                );
                let _ = tx.try_send(AssetEvent::Error(err.clone()));
                return Err(GeniusError::ManifestError(err).into());
            }
        } else {
            let err = format!("Model '{}' not found in registry", name);
            let _ = tx.try_send(AssetEvent::Error(err.clone()));
            return Err(GeniusError::ManifestError(err).into());
        };

        match spec.format {
            ModelFormat::Safetensors => {
                self.ensure_safetensors_model(name, &spec, tx, silent).await
            }
            ModelFormat::Gguf => {
                self.ensure_gguf_model(name, &spec, tx, silent).await
            }
        }
    }

    /// Download a single-file GGUF model.
    async fn ensure_gguf_model(
        &self,
        name: &str,
        spec: &ModelSpec,
        mut tx: mpsc::Sender<AssetEvent>,
        silent: bool,
    ) -> Result<PathBuf> {
        let cache_dir = self.registry.get_cache_dir();
        fs::create_dir_all(&cache_dir)?;

        let path = cache_dir.join(&spec.filename);
        if path.exists() {
            let _ = tx
                .send(AssetEvent::Complete(path.display().to_string()))
                .await;
            return Ok(path);
        }

        if !silent {
            println!("Downloading {} from {}...", spec.filename, spec.repo);
        }
        self.download_file_with_events(spec, &path, tx.clone())
            .await?;

        // If it was a new model (resolved via heuristic), record it
        if self.registry.resolve(name).is_none() {
            let mut registry = ModelRegistry::new()?;
            registry.record_model(ModelEntry {
                name: name.to_string(),
                repo: spec.repo.clone(),
                filename: spec.filename.clone(),
                quantization: spec.quantization.clone(),
                purpose: crate::registry::ModelPurpose::Inference,
                format: ModelFormat::Gguf,
                files: vec![],
            })?;
        }

        let _ = tx
            .send(AssetEvent::Complete(path.display().to_string()))
            .await;
        Ok(path)
    }

    /// Download a multi-file safetensors model directory.
    /// Downloads listed files + discovers weight shards from the index.
    /// Returns the model directory path.
    async fn ensure_safetensors_model(
        &self,
        name: &str,
        spec: &ModelSpec,
        mut tx: mpsc::Sender<AssetEvent>,
        silent: bool,
    ) -> Result<PathBuf> {
        let cache_dir = self.registry.get_cache_dir();
        let model_dir = cache_dir.join(&spec.filename);
        fs::create_dir_all(&model_dir)?;

        // Check if already downloaded (config.json exists as sentinel)
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            let _ = tx
                .send(AssetEvent::Complete(model_dir.display().to_string()))
                .await;
            return Ok(model_dir);
        }

        if !silent {
            println!("Downloading {} from {}...", spec.filename, spec.repo);
        }

        // Start with the files listed in the registry entry
        let mut files_to_download: Vec<String> = spec.files.clone();
        if files_to_download.is_empty() {
            // Minimum set for safetensors models
            files_to_download = vec![
                "config.json".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
            ];
        }

        // Download metadata files first
        for file in &files_to_download {
            let file_path = model_dir.join(file);
            if file_path.exists() {
                continue;
            }
            let file_spec = ModelSpec {
                repo: spec.repo.clone(),
                filename: file.clone(),
                quantization: spec.quantization.clone(),
                format: ModelFormat::Safetensors,
                files: vec![],
            };
            let _ = tx
                .try_send(AssetEvent::Started(format!("{}/{}", spec.repo, file)));
            self.download_file_with_events(&file_spec, &file_path, tx.clone())
                .await?;
        }

        // Discover safetensors shards from the index file
        let index_path = model_dir.join("model.safetensors.index.json");
        let shard_files = if index_path.exists() {
            Self::parse_safetensors_index(&index_path)?
        } else {
            // Single-file model: try model.safetensors
            vec!["model.safetensors".to_string()]
        };

        // Download weight shards
        let total_shards = shard_files.len();
        for (i, shard) in shard_files.iter().enumerate() {
            let shard_path = model_dir.join(shard);
            if shard_path.exists() {
                continue;
            }
            if !silent {
                println!("  [{}/{}] {}", i + 1, total_shards, shard);
            }
            let _ = tx.try_send(AssetEvent::Started(format!(
                "[{}/{}] {}",
                i + 1,
                total_shards,
                shard
            )));
            let shard_spec = ModelSpec {
                repo: spec.repo.clone(),
                filename: shard.clone(),
                quantization: spec.quantization.clone(),
                format: ModelFormat::Safetensors,
                files: vec![],
            };
            self.download_file_with_events(&shard_spec, &shard_path, tx.clone())
                .await?;
        }

        // Record in dynamic registry if not already known
        if self.registry.resolve(name).is_none() {
            let mut registry = ModelRegistry::new()?;
            registry.record_model(ModelEntry {
                name: name.to_string(),
                repo: spec.repo.clone(),
                filename: spec.filename.clone(),
                quantization: spec.quantization.clone(),
                purpose: crate::registry::ModelPurpose::Inference,
                format: ModelFormat::Safetensors,
                files: spec.files.clone(),
            })?;
        }

        let _ = tx
            .send(AssetEvent::Complete(model_dir.display().to_string()))
            .await;
        Ok(model_dir)
    }

    /// Parse a safetensors index file to discover weight shard filenames.
    fn parse_safetensors_index(index_path: &PathBuf) -> Result<Vec<String>> {
        let content = fs::read_to_string(index_path)?;
        // The index JSON has a "weight_map" key mapping layer names to shard files.
        // We extract unique shard filenames.
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("failed to parse safetensors index: {e}"))?;
        let weight_map = parsed
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("safetensors index missing weight_map"))?;

        let mut shards: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        shards.sort();
        shards.dedup();
        Ok(shards)
    }

    async fn download_file_with_events(
        &self,
        spec: &ModelSpec,
        final_path: &PathBuf,
        sender: mpsc::Sender<AssetEvent>,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );
        let _ = sender
            .clone()
            .try_send(AssetEvent::Started(format!("Downloading from: {}", url)));
        if !final_path.exists() {
            println!("DEBUG: Downloading from URL: {}", url);
        }

        let partial_path = final_path.with_extension("partial");
        let client = surf::Client::new().with(RedirectMiddleware::new(5));
        let response = client
            .get(&url)
            .await
            .map_err(|e| anyhow::anyhow!("Surf request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("Download failed with status: {}", status));
        }

        let total_size = response
            .header("Content-Length")
            .and_then(|h| h.last().as_str().parse::<u64>().ok())
            .unwrap_or(0);

        let mut reader = ProgressReader {
            inner: response,
            current: 0,
            total: total_size,
            sender,
        };

        {
            let std_file = std::fs::File::create(&partial_path)
                .map_err(|e| anyhow::anyhow!("Failed to create partial file: {}", e))?;
            let mut file: async_std::fs::File = std_file.into();

            if let Err(e) = futures::io::copy(&mut reader, &mut file).await {
                let _ = std::fs::remove_file(&partial_path);
                return Err(anyhow::anyhow!("Streaming failed: {}", e));
            }
        }

        if !partial_path.exists() {
            return Err(anyhow::anyhow!(
                "Partial file missing before rename: {:?}",
                partial_path
            ));
        }

        if let Err(e) = std::fs::rename(&partial_path, final_path) {
            eprintln!(
                "Warning: rename {:?} -> {:?} failed ({}), falling back to copy...",
                partial_path, final_path, e
            );
            std::fs::copy(&partial_path, final_path).map_err(|e| {
                anyhow::anyhow!("Failed to finalize model file (copy fallback): {}", e)
            })?;
            let _ = std::fs::remove_file(&partial_path);
        }
        Ok(())
    }
}

struct RedirectMiddleware {
    max_attempts: u8,
}

impl RedirectMiddleware {
    pub fn new(max_attempts: u8) -> Self {
        Self { max_attempts }
    }
}

#[surf::utils::async_trait]
impl surf::middleware::Middleware for RedirectMiddleware {
    async fn handle(
        &self,
        req: surf::Request,
        client: surf::Client,
        next: surf::middleware::Next<'_>,
    ) -> surf::Result<surf::Response> {
        let mut attempts = 0;
        let mut current_req = req;

        loop {
            // Check attempts
            if attempts > self.max_attempts {
                return Err(surf::Error::from_str(
                    surf::StatusCode::LoopDetected,
                    "Too many redirects",
                ));
            }

            // Clone req for the attempt (body might be an issue if not reusable, but for GET it's fine)
            // surf::Request cloning is usually cheap (Arc-ish for body?).
            // Wait, Request isn't trivially cloneable if body is a naive stream.
            // But `current_req.clone()` works in surf.
            let req_clone = current_req.clone();

            let response = next.run(req_clone, client.clone()).await?;

            if response.status().is_redirection() {
                if let Some(location) = response.header("Location") {
                    let loc_str = location.last().as_str().to_string();
                    // Update URL
                    // Use Url parsing to handle relative redirects?
                    // For HF, usually absolute.
                    // I will assume absolute or handle simple parse.

                    let new_url = match surf::Url::parse(&loc_str) {
                        Ok(u) => u,
                        Err(_) => {
                            // Try joining with base?
                            let base = current_req.url();
                            match base.join(&loc_str) {
                                Ok(u) => u,
                                Err(_) => {
                                    return Err(surf::Error::from_str(
                                        surf::StatusCode::BadGateway,
                                        "Invalid redirect location",
                                    ))
                                }
                            }
                        }
                    };

                    current_req = surf::Request::new(current_req.method(), new_url);
                    // Copy headers? usually yes.
                    // For now, new request is clean. simple GET.
                    // HF auth headers not needed for public models, but if they were, we'd copy.

                    attempts += 1;
                    continue;
                }
            }

            return Ok(response);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[async_std::test]
    async fn test_ensure_model_tiny() {
        let authority = AssetAuthority::new().unwrap();
        // Use a temp dir for testing if possible, but for now we'll just test the resolve logic
        // and assume connectivity is allowed in this environment.
        let name = "tiny-model";
        let res = authority.ensure_model(name).await;
        assert!(
            res.is_ok(),
            "Should resolve and download (or find) tiny-model"
        );
        let path = res.unwrap();
        assert!(path.exists());
    }

    #[async_std::test]
    async fn test_ensure_model_stream() {
        let authority = AssetAuthority::new().unwrap();
        let name = "tiny-model";

        let mut rx = authority.ensure_model_stream(name);
        let mut saw_started = false;
        let mut saw_complete = false;

        while let Some(event) = rx.next().await {
            match event {
                AssetEvent::Started(_) => saw_started = true,
                AssetEvent::Complete(p) => {
                    saw_complete = true;
                    assert!(
                        std::path::Path::new(&p).exists(),
                        "Complete path must exist"
                    );
                }
                AssetEvent::Error(e) => panic!("Download error: {}", e),
                _ => {}
            }
        }

        assert!(saw_started, "Should have received Started event");
        assert!(saw_complete, "Should have received Complete event");
    }
}

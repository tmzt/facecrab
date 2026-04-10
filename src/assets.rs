use crate::registry::ModelEntry;
use crate::registry::ModelRegistry;
use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use crate::core_types::{ModelFormat, ModelSpec, AssetEvent, GeniusError};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::Xxh3;

const FACECRAB_JSON: &str = "facecrab.json";
const HASH_CHUNK: usize = 64 * 1024;

/// Compute xxh3 hash of a file, returned as lowercase hex.
fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Xxh3::new();
    let mut buf = vec![0u8; HASH_CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:016x}", hasher.digest()))
}

/// Write a `facecrab.json` manifest for the given files into `dir`.
fn write_manifest(dir: &Path, files: &[impl AsRef<Path>]) -> Result<()> {
    let mut map: HashMap<String, String> = HashMap::new();
    for f in files {
        let path = dir.join(f.as_ref());
        if path.exists() {
            let key = f.as_ref().to_string_lossy().to_string();
            map.insert(key, hash_file(&path)?);
        }
    }
    let manifest = serde_json::json!({ "version": 1, "files": map });
    fs::write(dir.join(FACECRAB_JSON), serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

/// Verify that all files listed in `facecrab.json` exist and match their stored hashes.
/// Returns Ok(true) if the manifest is present and all hashes match,
/// Ok(false) if the manifest is missing or any hash mismatches,
/// Err if the manifest is malformed.
fn verify_manifest(dir: &Path) -> Result<bool> {
    let manifest_path = dir.join(FACECRAB_JSON);
    if !manifest_path.exists() {
        return Ok(false);
    }
    let content = fs::read_to_string(&manifest_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;
    let files = match parsed.get("files").and_then(|v| v.as_object()) {
        Some(m) => m,
        None => return Ok(false),
    };
    if files.is_empty() {
        return Ok(false);
    }
    for (name, stored) in files {
        let path = dir.join(name);
        if !path.exists() {
            log::debug!("facecrab: missing {name}");
            return Ok(false);
        }
        let actual = hash_file(&path)?;
        let expected = stored.as_str().unwrap_or("");
        if actual != expected {
            log::warn!("facecrab: hash mismatch for {name}: got {actual}, expected {expected}");
            return Ok(false);
        }
    }
    Ok(true)
}

/// Verify a single-file manifest (GGUF sidecar): file hash must match the stored value.
fn verify_manifest_single(file: &Path, manifest_path: &Path) -> Result<bool> {
    let content = fs::read_to_string(manifest_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;
    let files = match parsed.get("files").and_then(|v| v.as_object()) {
        Some(m) => m,
        None => return Ok(false),
    };
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    let expected = match files.get(name.as_ref()).and_then(|v| v.as_str()) {
        Some(h) => h,
        None => return Ok(false),
    };
    let actual = hash_file(file)?;
    if actual != expected {
        log::warn!("facecrab: hash mismatch for {name}: got {actual}, expected {expected}");
        return Ok(false);
    }
    Ok(true)
}

// ── Adaptive concurrency ──────────────────────────────────────────────────

const MAX_SHARD_CONCURRENCY: usize = 4;

/// Tracks per-shard throughput and adjusts the concurrency level heuristically.
///
/// Strategy:
/// - Start at `max` concurrent downloads.
/// - After each shard completes, record its bytes/sec.
/// - If the latest throughput drops below 50% of the running peak and we
///   haven't changed concurrency in the last 5 s, back off by 1.
/// - If the latest throughput recovers above 85% of peak, try adding 1 back
///   (up to `max`).
struct AdaptiveConcurrency {
    current: usize,
    max: usize,
    peak_bps: f64,
    last_adjust: std::time::Instant,
}

impl AdaptiveConcurrency {
    fn new(max: usize) -> Self {
        Self {
            current: max,
            max,
            peak_bps: 0.0,
            last_adjust: std::time::Instant::now(),
        }
    }

    /// Record a completed shard and return the (possibly updated) concurrency.
    fn record(&mut self, bytes: u64, secs: f64) -> usize {
        if secs < 0.1 {
            return self.current; // too quick to measure meaningfully
        }
        let bps = bytes as f64 / secs;
        if bps > self.peak_bps {
            self.peak_bps = bps;
        }

        // Rate-limit adjustments to once every 5 s.
        if self.last_adjust.elapsed().as_secs_f64() < 5.0 || self.peak_bps <= 0.0 {
            return self.current;
        }

        let ratio = bps / self.peak_bps;
        if ratio < 0.50 && self.current > 1 {
            self.current -= 1;
            self.last_adjust = std::time::Instant::now();
            log::info!(
                "facecrab: throughput fell to {:.0}% of peak ({:.1} MB/s → {:.1} MB/s), concurrency ↓{}",
                ratio * 100.0, self.peak_bps / 1e6, bps / 1e6, self.current
            );
        } else if ratio > 0.85 && self.current < self.max {
            self.current += 1;
            self.last_adjust = std::time::Instant::now();
            log::info!(
                "facecrab: throughput healthy ({:.1} MB/s), concurrency ↑{}",
                bps / 1e6, self.current
            );
        }

        self.current
    }
}

// ── Core download primitive ───────────────────────────────────────────────

/// Download `url` to `final_path` (via a `.partial` staging file).
/// Sends `AssetEvent::Progress` updates through `sender`.
/// Returns the number of bytes written.
async fn download_url_to_file(
    url: &str,
    final_path: &Path,
    sender: mpsc::Sender<AssetEvent>,
) -> Result<u64> {
    let partial_path = final_path.with_extension("partial");
    let client = surf::Client::new().with(RedirectMiddleware::new(5));
    let mut request = client.get(url);

    let hf_token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| option_env!("HF_TOKEN_BAKED").map(|s| s.to_string()));
    if let Some(token) = hf_token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }

    log::debug!("facecrab: GET {url}");
    let response = request.await.map_err(|e| {
        // Classify connection errors so DNS failures are obvious in logcat.
        let msg = e.to_string();
        if msg.contains("address information")
            || msg.contains("Name or service")
            || msg.contains("No such host")
            || msg.contains("lookup")
        {
            log::error!("facecrab: DNS lookup failed for {url}: {e}");
            anyhow::anyhow!("DNS lookup failed for {url}: {e}")
        } else {
            log::error!("facecrab: connection error for {url}: {e}");
            anyhow::anyhow!("connection error for {url}: {e}")
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        log::error!("facecrab: HTTP {status} for {url}");
        return Err(anyhow::anyhow!("HTTP {status} for {url}"));
    }

    let content_length = response
        .header("Content-Length")
        .and_then(|h| h.last().as_str().parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = ProgressReader {
        inner: response,
        current: 0,
        total: content_length,
        sender,
    };

    {
        let std_file = std::fs::File::create(&partial_path)
            .map_err(|e| anyhow::anyhow!("create partial: {e}"))?;
        let mut file: async_std::fs::File = std_file.into();
        if let Err(e) = futures::io::copy(&mut reader, &mut file).await {
            let _ = std::fs::remove_file(&partial_path);
            log::error!("facecrab: stream error for {url}: {e}");
            return Err(anyhow::anyhow!("stream failed: {e}"));
        }
    }

    let bytes_written = reader.current;

    // Detect silent truncation: server closed the connection before sending
    // all bytes promised by Content-Length. futures::io::copy treats early EOF
    // as success, so we must check manually.
    if content_length > 0 && bytes_written < content_length {
        let _ = std::fs::remove_file(&partial_path);
        log::error!(
            "facecrab: truncated download for {url}: got {bytes_written}/{content_length} bytes"
        );
        return Err(anyhow::anyhow!(
            "truncated: {bytes_written}/{content_length} bytes for {url}"
        ));
    }

    if !partial_path.exists() {
        return Err(anyhow::anyhow!("partial file missing before rename: {partial_path:?}"));
    }
    if let Err(e) = std::fs::rename(&partial_path, final_path) {
        std::fs::copy(&partial_path, final_path)
            .map_err(|e| anyhow::anyhow!("finalize failed: {e}"))?;
        let _ = std::fs::remove_file(&partial_path);
        let _ = e;
    }

    log::debug!("facecrab: wrote {bytes_written} bytes → {final_path:?}");
    Ok(bytes_written)
}

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

    /// Create an authority with only a cache directory (no config loading).
    pub fn with_cache_dir(cache_dir: std::path::PathBuf) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::with_cache_dir(cache_dir)?,
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
    /// Uses the same cache directory as the parent authority.
    pub fn ensure_spec_stream(&self, spec: ModelSpec) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);
        let cache_dir = self.registry.get_cache_dir();

        async_std::task::spawn(async move {
            let mut err_tx = tx.clone();
            let result: Result<()> = async {
                let auth = AssetAuthority::with_cache_dir(cache_dir)?;
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

        // Verify via manifest if the file already exists.
        if path.exists() {
            let manifest_path = path.with_file_name(
                format!("{}.facecrab.json", spec.filename)
            );
            let cached = if manifest_path.exists() {
                match verify_manifest_single(&path, &manifest_path) {
                    Ok(ok) => ok,
                    Err(e) => { log::warn!("facecrab: manifest verify error for {name}: {e}"); false }
                }
            } else {
                false
            };
            if cached {
                let _ = tx
                    .send(AssetEvent::Complete(path.display().to_string()))
                    .await;
                return Ok(path);
            }
        }

        if !silent {
            println!("Downloading {} from {}...", spec.filename, spec.repo);
        }
        self.download_file_with_events(spec, &path, tx.clone())
            .await?;

        // Write a sidecar manifest for the GGUF file.
        {
            let manifest_path = path.with_file_name(
                format!("{}.facecrab.json", spec.filename)
            );
            if let Ok(hash) = hash_file(&path) {
                let manifest = serde_json::json!({ "version": 1, "files": { &spec.filename: hash } });
                if let Err(e) = fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap_or_default()) {
                    log::warn!("facecrab: failed to write manifest for {name}: {e}");
                }
            }
        }

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

        // Check if already fully downloaded by verifying facecrab.json hashes.
        match verify_manifest(&model_dir) {
            Ok(true) => {
                let _ = tx
                    .send(AssetEvent::Complete(model_dir.display().to_string()))
                    .await;
                return Ok(model_dir);
            }
            Ok(false) => {}
            Err(e) => log::warn!("facecrab: manifest verify error for {name}: {e}"),
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

        // Download weight shards in parallel with adaptive concurrency.
        {
            let total_shards = shard_files.len();
            let mut adapt = AdaptiveConcurrency::new(MAX_SHARD_CONCURRENCY);

            // Only enqueue shards that aren't already on disk.
            let mut queue: VecDeque<(usize, String)> = shard_files
                .iter()
                .enumerate()
                .filter(|(_, s)| !model_dir.join(s).exists())
                .map(|(i, s)| (i, s.clone()))
                .collect();

            if !queue.is_empty() {
                type ShardFut = std::pin::Pin<Box<dyn futures::Future<Output = Result<(usize, u64, f64)>> + Send>>;
                let mut in_flight: FuturesUnordered<ShardFut> = FuturesUnordered::new();

                macro_rules! push_next {
                    () => {
                        if let Some((i, shard)) = queue.pop_front() {
                            let shard_path = model_dir.join(&shard);
                            let url = format!(
                                "https://huggingface.co/{}/resolve/main/{}",
                                spec.repo, shard
                            );
                            if !silent {
                                println!("  [{}/{}] {}", i + 1, total_shards, shard);
                            }
                            let _ = tx.try_send(AssetEvent::Started(format!(
                                "[{}/{}] {}", i + 1, total_shards, shard
                            )));
                            let tx2 = tx.clone();
                            in_flight.push(Box::pin(async move {
                                let t0 = std::time::Instant::now();
                                let bytes = download_url_to_file(&url, &shard_path, tx2).await?;
                                Ok((i, bytes, t0.elapsed().as_secs_f64()))
                            }));
                        }
                    };
                }

                // Seed initial batch.
                for _ in 0..adapt.current.min(queue.len()) {
                    push_next!();
                }

                // Drain completions, refill up to current concurrency.
                while let Some(result) = in_flight.next().await {
                    let (_, bytes, secs) = result?;
                    let concurrency = adapt.record(bytes, secs);
                    while in_flight.len() < concurrency && !queue.is_empty() {
                        push_next!();
                    }
                }
            }
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

        // Write facecrab.json: hash all downloaded files (spec files + shards).
        {
            let mut all_files: Vec<String> = spec.files.clone();
            for s in &shard_files {
                if !all_files.contains(s) {
                    all_files.push(s.clone());
                }
            }
            if let Err(e) = write_manifest(&model_dir, &all_files) {
                log::warn!("facecrab: failed to write manifest for {name}: {e}");
            }
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
        download_url_to_file(&url, final_path, sender).await?;
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

use crate::registry::ModelEntry;
use crate::registry::ModelRegistry;
use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use crate::core_types::{ModelFormat, ModelSpec, AssetEvent, GeniusError};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

impl AssetAuthority {
    #[cfg(feature = "registry")]
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
        let cache_dir = self.registry.get_cache_dir();
        let registry_models = self.registry.list_models();
        let name = name.to_string();

        smol::spawn(async move {
            let result: Result<()> = async {
                let auth = AssetAuthority::with_cache_dir(cache_dir)?;
                auth.ensure_model_internal(&name, tx, true).await?;
                Ok(())
            }.await;
            if let Err(e) = result {
                log::error!("ensure_model error: {e}");
            }
        }).detach();

        let mut path = None;
        while let Some(event) = rx.next().await {
            if let AssetEvent::Complete(p) = event {
                path = Some(PathBuf::from(p));
            }
        }
        path.ok_or_else(|| anyhow::anyhow!("no path returned"))
    }

    /// Download a model and return a stream of [AssetEvent]s.
    pub fn ensure_model_stream(&self, name: &str) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);
        let cache_dir = self.registry.get_cache_dir();
        let name = name.to_string();

        smol::spawn(async move {
            let mut err_tx = tx.clone();
            let result: Result<()> = async {
                let auth = AssetAuthority::with_cache_dir(cache_dir)?;
                auth.ensure_model_internal(&name, tx, false).await?;
                Ok(())
            }.await;
            if let Err(e) = result {
                let _ = err_tx.send(AssetEvent::Error(e.to_string())).await;
            }
        }).detach();

        rx
    }

    /// Download a model from a spec and return a stream of [AssetEvent]s.
    pub fn ensure_spec_stream(&self, spec: ModelSpec) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);
        let cache_dir = self.registry.get_cache_dir();

        smol::spawn(async move {
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
            }.await;
            if let Err(e) = result {
                let _ = err_tx.send(AssetEvent::Error(e.to_string())).await;
            }
        }).detach();

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
                let err = format!("Model '{}' not found and invalid format", name);
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
            let _ = tx.send(AssetEvent::Complete(path.display().to_string())).await;
            return Ok(path);
        }

        if !silent {
            log::info!("Downloading {} from {}...", spec.filename, spec.repo);
        }
        self.download_file_with_events(spec, &path, tx.clone()).await?;

        let _ = tx.send(AssetEvent::Complete(path.display().to_string())).await;
        Ok(path)
    }

    /// Download a multi-file safetensors model directory.
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
            let _ = tx.send(AssetEvent::Complete(model_dir.display().to_string())).await;
            return Ok(model_dir);
        }

        if !silent {
            log::info!("Downloading {} from {}...", spec.filename, spec.repo);
        }

        // Start with the files listed in the spec
        let mut files_to_download: Vec<String> = spec.files.clone();
        if files_to_download.is_empty() {
            files_to_download = vec![
                "config.json".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
            ];
        }

        // Download metadata files first
        for file in &files_to_download {
            let file_path = model_dir.join(file);
            if file_path.exists() { continue; }
            let file_spec = ModelSpec {
                repo: spec.repo.clone(),
                filename: file.clone(),
                quantization: spec.quantization.clone(),
                format: ModelFormat::Safetensors,
                files: vec![],
            };
            let _ = tx.try_send(AssetEvent::Started(format!("{}/{}", spec.repo, file)));
            self.download_file_with_events(&file_spec, &file_path, tx.clone()).await?;
        }

        // Discover safetensors shards from the index file
        let index_path = model_dir.join("model.safetensors.index.json");
        let shard_files = if index_path.exists() {
            Self::parse_safetensors_index(&index_path)?
        } else {
            vec!["model.safetensors".to_string()]
        };

        // Download weight shards
        let total_shards = shard_files.len();
        for (i, shard) in shard_files.iter().enumerate() {
            let shard_path = model_dir.join(shard);
            if shard_path.exists() { continue; }
            let _ = tx.try_send(AssetEvent::Started(format!("[{}/{}] {}", i + 1, total_shards, shard)));
            let shard_spec = ModelSpec {
                repo: spec.repo.clone(),
                filename: shard.clone(),
                quantization: spec.quantization.clone(),
                format: ModelFormat::Safetensors,
                files: vec![],
            };
            self.download_file_with_events(&shard_spec, &shard_path, tx.clone()).await?;
        }

        let _ = tx.send(AssetEvent::Complete(model_dir.display().to_string())).await;
        Ok(model_dir)
    }

    /// Parse a safetensors index file to discover weight shard filenames.
    fn parse_safetensors_index(index_path: &PathBuf) -> Result<Vec<String>> {
        let content = fs::read_to_string(index_path)?;
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

    /// Download a single file from HuggingFace with progress events.
    /// Uses smol + rustls — no curl, no openssl, no system SSL.
    async fn download_file_with_events(
        &self,
        spec: &ModelSpec,
        final_path: &PathBuf,
        mut sender: mpsc::Sender<AssetEvent>,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );
        log::debug!("downloading: {}", url);

        let partial_path = final_path.with_extension("partial");

        // Follow redirects (up to 5)
        let (status, headers, body) = http_get_follow(&url, 5).await
            .map_err(|e| anyhow::anyhow!("download failed: {e}"))?;

        if status != 200 {
            return Err(anyhow::anyhow!("HTTP {status} for {url}"));
        }

        let total_size: u64 = headers.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(0);

        // Write body to partial file with progress
        let mut file = std::fs::File::create(&partial_path)
            .map_err(|e| anyhow::anyhow!("create file: {e}"))?;
        let mut written: u64 = 0;
        let chunk_size = 64 * 1024; // 64KB progress updates
        let mut pos = 0;

        while pos < body.len() {
            let end = (pos + chunk_size).min(body.len());
            file.write_all(&body[pos..end])?;
            written += (end - pos) as u64;
            pos = end;
            let _ = sender.try_send(AssetEvent::Progress(written, total_size));
        }
        drop(file);

        // Rename partial → final
        if let Err(e) = std::fs::rename(&partial_path, final_path) {
            std::fs::copy(&partial_path, final_path)
                .map_err(|e| anyhow::anyhow!("finalize failed: {e}"))?;
            let _ = std::fs::remove_file(&partial_path);
        }

        Ok(())
    }
}

// ── Minimal HTTP client using smol + rustls ─────────────────────────────

/// Perform an HTTPS GET with redirect following. Returns (status, headers, body).
async fn http_get_follow(url: &str, max_redirects: u8) -> Result<(u16, Vec<(String, String)>, Vec<u8>)> {
    let mut current_url = url.to_string();

    for _ in 0..=max_redirects {
        let (status, headers, body) = http_get(&current_url).await?;

        if (301..=308).contains(&status) {
            if let Some((_, location)) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("location")) {
                current_url = if location.starts_with("http") {
                    location.clone()
                } else {
                    // Relative redirect — join with base origin
                    let (scheme, rest) = current_url.split_once("://").unwrap_or(("https", &current_url));
                    let host_part = rest.split('/').next().unwrap_or(rest);
                    format!("{scheme}://{host_part}{location}")
                };
                continue;
            }
        }

        return Ok((status, headers, body));
    }

    Err(anyhow::anyhow!("too many redirects"))
}

/// Single HTTPS GET request using smol::net + rustls.
async fn http_get(url: &str) -> Result<(u16, Vec<(String, String)>, Vec<u8>)> {
    let (scheme, rest) = url.split_once("://").ok_or_else(|| anyhow::anyhow!("invalid URL: {url}"))?;
    let (host_port, path) = rest.split_once('/').unwrap_or((rest, ""));
    let path = format!("/{path}");
    let (host, port) = if let Some((h, p)) = host_port.split_once(':') {
        (h, p.parse::<u16>().unwrap_or(443))
    } else {
        (host_port, if scheme == "https" { 443 } else { 80 })
    };

    let addr = format!("{host}:{port}");
    let tcp = smol::net::TcpStream::connect(&addr).await
        .map_err(|e| anyhow::anyhow!("connect {addr}: {e}"))?;

    if scheme == "https" {
        // TLS via futures-rustls (re-exports rustls)
        use futures_rustls::rustls;

        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| anyhow::anyhow!("rustls config: {e}"))?
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let server_name = futures_rustls::pki_types::ServerName::try_from(host.to_string())
            .map_err(|e| anyhow::anyhow!("invalid server name: {e}"))?;

        let connector = futures_rustls::TlsConnector::from(Arc::new(config));
        let tls_stream = connector.connect(server_name, tcp).await
            .map_err(|e| anyhow::anyhow!("TLS handshake: {e}"))?;

        http_request_on_stream(tls_stream, host, &path).await
    } else {
        http_request_on_stream(tcp, host, &path).await
    }
}

/// Send HTTP/1.1 GET and read the response from any AsyncRead+AsyncWrite stream.
async fn http_request_on_stream<S>(mut stream: S, host: &str, path: &str) -> Result<(u16, Vec<(String, String)>, Vec<u8>)>
where
    S: smol::io::AsyncRead + smol::io::AsyncWrite + Unpin,
{
    use smol::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: facecrab/0.1\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).await?;
    stream.flush().await?;

    let mut reader = BufReader::new(stream);

    // Read status line
    let mut status_line = String::new();
    reader.read_line(&mut status_line).await?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Read headers
    let mut headers = Vec::new();
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let trimmed = line.trim();
        if trimmed.is_empty() { break; }
        if let Some((key, val)) = trimmed.split_once(':') {
            let k = key.trim().to_string();
            let v = val.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().ok();
            }
            if k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked") {
                chunked = true;
            }
            headers.push((k, v));
        }
    }

    // Read body
    let body = if chunked {
        read_chunked_body(&mut reader).await?
    } else if let Some(len) = content_length {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).await?;
        buf
    } else {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        buf
    };

    Ok((status, headers, body))
}

/// Read a chunked transfer-encoding body.
async fn read_chunked_body<R: smol::io::AsyncBufRead + Unpin>(reader: &mut R) -> Result<Vec<u8>> {
    use smol::io::{AsyncBufReadExt, AsyncReadExt};

    let mut body = Vec::new();
    loop {
        let mut size_line = String::new();
        reader.read_line(&mut size_line).await?;
        let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
        if size == 0 { break; }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk).await?;
        body.extend_from_slice(&chunk);
        // Read trailing \r\n
        let mut crlf = [0u8; 2];
        reader.read_exact(&mut crlf).await?;
    }
    Ok(body)
}

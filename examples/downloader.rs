use facecrab::AssetAuthority;
use futures::StreamExt;
use rusty_genius_core::protocol::AssetEvent;
use std::error::Error;

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let authority = AssetAuthority::new()?;
    let model_name = "tiny-model";

    println!("--- Mode 1: Simple One-Shot ---");
    println!("Checking model: {}", model_name);
    let path = authority.ensure_model(model_name).await?;
    println!("Model ready at: {:?}\n", path);

    println!("--- Mode 2: Event-Based Streaming ---");
    // We'll delete the file first to force a download progress for demonstration
    if path.exists() {
        println!("Cleaning up cached model for demonstration...");
        std::fs::remove_file(&path)?;
    }

    let mut events = authority.ensure_model_stream(model_name);
    while let Some(event) = events.next().await {
        match event {
            AssetEvent::Started(name) => println!("Started resolution for: {}", name),
            AssetEvent::Progress(current, total) => {
                let pct = if total > 0 {
                    (current as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                print!("\rDownload Progress: {:.1}% ({}/{})", pct, current, total);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            AssetEvent::Complete(path) => {
                println!("\nSuccessfully completed: {}", path);
            }
            AssetEvent::Error(err) => {
                eprintln!("\nAsset Error: {}", err);
            }
        }
    }

    Ok(())
}

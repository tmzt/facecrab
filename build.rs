fn main() {
    println!("cargo:rerun-if-env-changed=HF_TOKEN");
    if let Ok(token) = std::env::var("HF_TOKEN") {
        println!("cargo:rustc-env=HF_TOKEN_BAKED={token}");
    }
}

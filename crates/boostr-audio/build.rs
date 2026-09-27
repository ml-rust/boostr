//! Build script for boostr-audio.
//!
//! `g2p-neural` embeds the g2p_lm checkpoint via `include_bytes!`/
//! `include_str!` in `src/g2p/neural/embedded.rs`. Those assets
//! (`assets/g2p_lm/model.safetensors.zlib`, `config.json`, `vocab.json`) do
//! not exist until `assets/g2p_lm/pack_model.py` has packed a trained
//! checkpoint, so this checks for them up front and fails with the exact
//! command to run, rather than letting `include_bytes!` fail with a bare
//! "file not found" pointing at a macro instead of a fix.

use std::path::Path;

fn main() {
    println!("cargo::rerun-if-changed=assets/g2p_lm");
    if std::env::var_os("CARGO_FEATURE_G2P_NEURAL").is_none() {
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let assets_dir = Path::new(&manifest_dir).join("assets/g2p_lm");
    let required = ["model.safetensors.zlib", "config.json", "vocab.json"];
    let missing: Vec<&str> = required
        .iter()
        .filter(|name| !assets_dir.join(name).exists())
        .copied()
        .collect();

    if !missing.is_empty() {
        panic!(
            "g2p-neural: missing embedded asset(s) in {}: {}. Train g2p_lm \
             (oxidizr/models/g2p-char/g2p-char.yaml), then run:\n\
             \n    python3 crates/boostr-audio/assets/g2p_lm/pack_model.py \\\n        \
             <checkpoint_dir> $BOOSTR_MODELS_DIR/g2p_lm/data/vocab.json \\\n        \
             crates/boostr-audio/assets/g2p_lm\n\
             \nfrom the boostr workspace root before building with --features g2p-neural.",
            assets_dir.display(),
            missing.join(", "),
        );
    }
}

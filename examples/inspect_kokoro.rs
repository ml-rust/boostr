//! One-off utility: parse the upstream `kokoro-v1_0.pth` and print its
//! state-dict key layout. Useful for sanity-checking the tier-3 loader
//! against a live checkpoint.
//!
//! Usage:
//!   cargo run --example inspect_kokoro -- /path/to/Kokoro-82M/kokoro-v1_0.pth

use boostr::format::TorchStateDict;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_kokoro PATH/TO/kokoro-v1_0.pth")?;
    let sd = TorchStateDict::open(&path)?;
    let mut keys: Vec<&str> = sd.keys().collect();
    keys.sort();
    println!("total tensors: {}", keys.len());
    for k in keys.iter().take(30) {
        println!("  {k}");
    }
    println!("  ...");
    for k in keys.iter().rev().take(10).rev() {
        println!("  {k}");
    }

    // Sanity-check that the keys we hardcoded in load_kokoro_v2 exist.
    let expected_samples = [
        "bert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "predictor.text_encoder.lstms.0.weight_ih_l0",
        "predictor.duration_proj.linear_layer.weight",
        "text_encoder.embedding.weight",
        "decoder.generator.conv_post.parametrizations.weight.original0",
        "decoder.generator.m_source.l_linear.weight",
    ];
    println!("\nsanity check:");
    for k in &expected_samples {
        let found = sd.has(k);
        println!("  {} {k}", if found { "✓" } else { "✗" });
    }
    Ok(())
}

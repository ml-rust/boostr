//! Verify VoxCPM2's text tokenizer, loaded through splintr's generic
//! HuggingFace `tokenizer.json` reader, against the reference
//! `tokenizer.tokenize(text)` -> `convert_tokens_to_ids(tokens)` path.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_tokenizer_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`tokenizer.json` at its root).
//! `FIXTURE_DIR` holds `tokenizer_fixture.safetensors` (int64 `ids_plain_{i}`
//! / `ids_expanded_{i}` for i in 0..8) and the sibling `tokenizer_fixture.json`
//! (each index's exact text and label), both written by
//! `audio/pipeline/make_tokenizer_fixture.py` from the reference.
//!
//! This is an EXACT integer-id gate, not a tolerance comparison like the
//! sibling tensor checks: one wrong id shifts every position after it in the
//! causal context, so there is no meaningful "close enough".
//!
//! ## Why `ids_plain`, never `ids_expanded`
//!
//! The reference model calls `tokenizer.tokenize` + `convert_tokens_to_ids`
//! directly, which never invokes the `TemplateProcessing` post-processor
//! (that only runs under `encode()`/`__call__`, confirmed by the fixture
//! script: `encode(text0) == [bos_id] + ids_plain(text0)` exactly) and never
//! runs `VoxCPM2Tokenizer._expand_ids`'s CJK multi-character split (that is a
//! separate wrapper the reference model does not use). So `ids_plain` is the
//! ground truth this gate checks against, for every text including the CJK
//! one. `ids_expanded` differs from `ids_plain` ONLY for the CJK text (index
//! 6, `cjk_multichar_split`) and this port deliberately does not implement
//! that split (see `boostr::model::audio::voxcpm::tokenizer` doc comments),
//! so index 6 is asserted to match `ids_plain`, not `ids_expanded`.
use std::collections::BTreeMap;
use std::path::PathBuf;

use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::load_tokenizer;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use serde::Deserialize;

#[derive(Deserialize)]
struct FixtureRecord {
    label: String,
    text: String,
    ids_plain: Vec<i64>,
    #[allow(dead_code)] // read for documentation purposes only, see module docs
    ids_expanded: Vec<i64>,
}

/// Compare `got` to `want` id-for-id. Returns `true` on an exact match; on a
/// mismatch, prints the label, the text, both sequences, and the index of the
/// first divergence.
fn check_ids(label: &str, text: &str, got: &[i64], want: &[i64]) -> bool {
    if got == want {
        println!("{label}: OK ({} ids)", got.len());
        return true;
    }
    let first_diff = got
        .iter()
        .zip(want.iter())
        .position(|(g, w)| g != w)
        .unwrap_or_else(|| got.len().min(want.len()));
    println!("{label}: MISMATCH");
    println!("  text: {text:?}");
    println!("  got  ({}): {got:?}", got.len());
    println!("  want ({}): {want:?}", want.len());
    println!("  first differing index: {first_diff}");
    false
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();

    let tokenizer = load_tokenizer(ck.join("tokenizer.json"))?;

    let json_text = std::fs::read_to_string(fx_dir.join("tokenizer_fixture.json"))?;
    let records: BTreeMap<String, FixtureRecord> = serde_json::from_str(&json_text)?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("tokenizer_fixture.safetensors"))?;

    let mut ok = true;
    for i in 0..8usize {
        let key = i.to_string();
        let record = records
            .get(&key)
            .unwrap_or_else(|| panic!("tokenizer_fixture.json missing index {i}"));

        let want_tensor = fx.load_tensor::<CpuRuntime>(&format!("ids_plain_{i}"), &device)?;
        let want: Vec<i64> = want_tensor.contiguous()?.to_vec();

        // The generator writes the same ids into both the safetensors and the
        // sibling JSON. If they ever disagree the fixture is internally
        // inconsistent and neither can be trusted as the reference, so fail
        // here rather than silently gating against whichever one was loaded.
        assert_eq!(
            record.ids_plain, want,
            "{}: tokenizer_fixture.json and .safetensors disagree on ids_plain_{i}",
            record.label
        );

        let got_u32 = tokenizer.encode_raw(&record.text);
        let got: Vec<i64> = got_u32.iter().map(|&id| i64::from(id)).collect();

        let pass = check_ids(&record.label, &record.text, &got, &want);
        ok &= pass;

        if record.label == "cjk_multichar_split" {
            println!(
                "  note: {} is deliberately checked against ids_plain, not \
                 ids_expanded -- the CJK multi-character split is out of \
                 scope for this port (see module docs)",
                record.label
            );
        }

        // A future post-processor regression (accidentally routing through
        // `encode` instead of `encode_raw`) would prepend BOS id 1 to every
        // sequence. Catch that loudly wherever the reference itself does not
        // start with 1.
        if want.first() != Some(&1) {
            let starts_with_bos = got.first() == Some(&1);
            if starts_with_bos {
                println!(
                    "{}: MISMATCH -- produced ids begin with BOS id 1 but the \
                     reference does not (post-processor template leaking in)",
                    record.label
                );
            }
            ok &= !starts_with_bos;
        }
    }

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}

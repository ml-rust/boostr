//! Numerical parity for [`VitsTextEncoder`] against HuggingFace's
//! `VitsTextEncoder` (`facebook/mms-tts-zlm`).
//!
//! Fixtures come from `tests/tools/vits_reference_dump/vits_reference_dump.py`,
//! which loads the real checkpoint, runs a few Malay sentences through the
//! whole model, and dumps every intermediate (this test uses only the text
//! encoder's three outputs). NOT checked in (torch/transformers plus the
//! checkpoint), so this skips unless BOTH are present:
//!
//! * the checkpoint — `MMS_TTS_ZLM_DIR`, else
//!   `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`
//! * the dumps — `VITS_REF_DIR`, else `$BOOSTR_MODELS_DIR/vits_reference/mms-tts-zlm`
//!
//! What this pins that unit tests cannot: the embedding scale
//! (`* sqrt(hidden_size)`), the relative-position attention window read from
//! the real `config.json`, and the exact `project` conv split point against
//! real trained weights (a unit test's hand-built tiny weights cannot catch a
//! transposed or off-by-one split).

use boostr::model::audio::vits::VitsTextEncoder;
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::{model_fixture, skip_notice};

/// The three cases measured 4.2e-6, 5.1e-6 and 6.8e-6 max abs diff — f32
/// rounding across ~192-wide dot products, nothing structural. Keeps ~15x
/// headroom: a wrong split point, a missing embedding scale, or a wrong
/// attention window would all move this by 1e-2 or more (see the module
/// docs), so anything above 1e-4 means a formula is wrong, not that the
/// tolerance is tight.
const TOL: f32 = 1e-4;

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn check(name: &str, ours: &Tensor<CpuRuntime>, reference: &Tensor<CpuRuntime>) {
    assert_eq!(
        ours.shape(),
        reference.shape(),
        "{name}: shape mismatch, ours {:?} reference {:?}",
        ours.shape(),
        reference.shape()
    );
    let a: Vec<f32> = ours.contiguous().unwrap().to_vec();
    let b: Vec<f32> = reference.contiguous().unwrap().to_vec();
    let (worst, at) = max_abs_diff(&a, &b);
    eprintln!(
        "{name}: max abs diff {worst:e} at flat index {at} (ours {}, reference {})",
        a[at], b[at]
    );
    assert!(worst < TOL, "{name}: max abs diff {worst:e} >= {TOL:e}");
}

struct Fixtures {
    checkpoint_dir: PathBuf,
    ref_dir: PathBuf,
}

fn fixtures() -> Option<Fixtures> {
    let checkpoint_dir = model_fixture("MMS_TTS_ZLM_DIR", "facebook/mms-tts-zlm")?;
    let ref_dir = model_fixture("VITS_REF_DIR", "vits_reference/mms-tts-zlm")?;
    Some(Fixtures {
        checkpoint_dir,
        ref_dir,
    })
}

#[test]
fn vits_text_encoder_matches_reference_for_every_sentence() {
    let Some(fx) = fixtures() else {
        skip_notice("vits reference dump", "VITS_REF_DIR");
        return;
    };

    let index_bytes = std::fs::read(fx.ref_dir.join("index.json"))
        .unwrap_or_else(|e| panic!("reading index.json: {e}"));
    let index: serde_json::Value =
        serde_json::from_slice(&index_bytes).expect("index.json is valid JSON");
    let sentences = index["sentences"]
        .as_array()
        .expect("index.json `sentences` is an array");
    assert!(!sentences.is_empty(), "index.json lists no sentences");

    let config_path = fx.checkpoint_dir.join("config.json");
    let config = VitsConfig::from_json_file(&config_path)
        .unwrap_or_else(|e| panic!("parsing {}: {e}", config_path.display()));
    config
        .validate()
        .expect("checkpoint config passes validation");

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let weights_path = fx.checkpoint_dir.join("model.safetensors");
    let mut weights = VarMap::<CpuRuntime>::from_safetensors(&weights_path, &device)
        .unwrap_or_else(|e| panic!("loading {}: {e}", weights_path.display()));
    let mut vb = VarBuilder::new(&mut weights, &device);
    let mut text_encoder_vb = vb.pp("text_encoder");
    let text_encoder = VitsTextEncoder::from_varbuilder(&mut text_encoder_vb, &config)
        .expect("loading VitsTextEncoder from the real checkpoint");
    drop(text_encoder_vb);

    for sentence in sentences {
        let file = sentence["file"]
            .as_str()
            .expect("sentence entry has `file`");
        let text = sentence["text"].as_str().unwrap_or("<unknown>");
        let fixture_path = fx.ref_dir.join(file);
        let mut fixture = VarMap::<CpuRuntime>::from_safetensors(&fixture_path, &device)
            .unwrap_or_else(|e| panic!("loading {}: {e}", fixture_path.display()));

        let input_ids = fixture
            .take_tensor("input_ids")
            .expect("fixture has input_ids");
        let attention_mask = fixture
            .take_tensor("attention_mask")
            .expect("fixture has attention_mask");
        let seq_len = attention_mask.shape()[1];
        let padding_mask = attention_mask
            .reshape(&[1, seq_len, 1])
            .expect("attention_mask reshapes to [1, seq, 1]");

        let out = text_encoder
            .forward(&client, &input_ids, &padding_mask, Some(&attention_mask))
            .unwrap_or_else(|e| panic!("{text}: VitsTextEncoder::forward failed: {e}"));

        let ref_last_hidden = fixture
            .take_tensor("text_encoder_last_hidden_state")
            .expect("fixture has text_encoder_last_hidden_state");
        let ref_prior_means = fixture
            .take_tensor("text_encoder_prior_means")
            .expect("fixture has text_encoder_prior_means");
        let ref_prior_log_variances = fixture
            .take_tensor("text_encoder_prior_log_variances")
            .expect("fixture has text_encoder_prior_log_variances");

        eprintln!("--- {text:?} ({seq_len} tokens) ---");
        check(
            "last_hidden_state",
            &out.last_hidden_state,
            &ref_last_hidden,
        );
        check("prior_means", &out.prior_means, &ref_prior_means);
        check(
            "prior_log_variances",
            &out.prior_log_variances,
            &ref_prior_log_variances,
        );
    }
}

//! Numerical parity for [`VitsStochasticDurationPredictor`]'s reverse path
//! and [`regulate_length`] against HuggingFace's `VitsModel`
//! (`facebook/mms-tts-zlm`).
//!
//! Fixtures come from `tests/tools/vits_reference_dump/vits_reference_dump.py`.
//! NOT checked in (torch/transformers plus the checkpoint), so this skips
//! unless BOTH are present:
//!
//! * the checkpoint — `MMS_TTS_ZLM_DIR`, else
//!   `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`
//! * the dumps — `VITS_REF_DIR`, else `$BOOSTR_MODELS_DIR/vits_reference/mms-tts-zlm`
//!
//! Two checks per sentence, run separately so [`regulate_length`]'s
//! exactness claim is not diluted by the duration predictor's float
//! rounding:
//!
//! 1. Our [`VitsStochasticDurationPredictor::forward_reverse_with_noise`],
//!    fed the fixture's text-encoder outputs and its captured
//!    `noise_randn_0` draw, against the fixture's `log_duration` (float
//!    rounding tolerance).
//! 2. Our [`regulate_length`], fed the fixture's OWN `log_duration` (not
//!    ours — this isolates the length-regulation arithmetic from unit 1's
//!    rounding), against the fixture's `duration`, `predicted_lengths`,
//!    `attn`, `expanded_prior_means`, `expanded_prior_log_variances`.

use boostr::model::audio::vits::{VitsStochasticDurationPredictor, regulate_length};
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::{model_fixture, skip_notice};

/// See `vits_text_encoder_parity.rs`'s `TOL` doc: f32 rounding across the
/// duration predictor's conv/flow stack, nothing structural. A wrong flow
/// order, a wrong noise scale, or a dropped conditioning term would move
/// this by 1e-2 or more.
const LOG_DURATION_TOL: f32 = 1e-4;
/// `regulate_length`'s float outputs (`expanded_prior_*`) are one masked
/// matmul away from the (exact) `attn`/`duration` inputs — f32 accumulation
/// only.
const EXPANDED_PRIOR_TOL: f32 = 1e-5;

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

fn check_close(name: &str, ours: &Tensor<CpuRuntime>, reference: &Tensor<CpuRuntime>, tol: f32) {
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
    assert!(worst < tol, "{name}: max abs diff {worst:e} >= {tol:e}");
}

fn check_exact_f32(name: &str, ours: &Tensor<CpuRuntime>, reference: &Tensor<CpuRuntime>) {
    assert_eq!(
        ours.shape(),
        reference.shape(),
        "{name}: shape mismatch, ours {:?} reference {:?}",
        ours.shape(),
        reference.shape()
    );
    let a: Vec<f32> = ours.contiguous().unwrap().to_vec();
    let b: Vec<f32> = reference.contiguous().unwrap().to_vec();
    assert_eq!(a, b, "{name}: expected bit-exact match");
}

fn check_exact_i64(name: &str, ours: &Tensor<CpuRuntime>, reference: &Tensor<CpuRuntime>) {
    assert_eq!(
        ours.shape(),
        reference.shape(),
        "{name}: shape mismatch, ours {:?} reference {:?}",
        ours.shape(),
        reference.shape()
    );
    let a: Vec<i64> = ours.contiguous().unwrap().to_vec();
    let b: Vec<i64> = reference.contiguous().unwrap().to_vec();
    assert_eq!(a, b, "{name}: expected bit-exact match");
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
fn vits_duration_regulation_matches_reference_for_every_sentence() {
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
    assert!(
        config.use_stochastic_duration_prediction,
        "facebook/mms-tts-zlm uses the stochastic duration predictor"
    );

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let weights_path = fx.checkpoint_dir.join("model.safetensors");
    let mut weights = VarMap::<CpuRuntime>::from_safetensors(&weights_path, &device)
        .unwrap_or_else(|e| panic!("loading {}: {e}", weights_path.display()));
    let mut vb = VarBuilder::new(&mut weights, &device);
    let mut duration_vb = vb.pp("duration_predictor");
    let duration_predictor =
        VitsStochasticDurationPredictor::from_varbuilder(&mut duration_vb, &config)
            .expect("loading VitsStochasticDurationPredictor from the real checkpoint");
    drop(duration_vb);

    for sentence in sentences {
        let file = sentence["file"]
            .as_str()
            .expect("sentence entry has `file`");
        let text = sentence["text"].as_str().unwrap_or("<unknown>");
        let fixture_path = fx.ref_dir.join(file);
        let mut fixture = VarMap::<CpuRuntime>::from_safetensors(&fixture_path, &device)
            .unwrap_or_else(|e| panic!("loading {}: {e}", fixture_path.display()));
        eprintln!("--- {text:?} ---");

        let last_hidden_state = fixture
            .take_tensor("text_encoder_last_hidden_state")
            .expect("fixture has text_encoder_last_hidden_state");
        let prior_means = fixture
            .take_tensor("text_encoder_prior_means")
            .expect("fixture has text_encoder_prior_means");
        let prior_log_variances = fixture
            .take_tensor("text_encoder_prior_log_variances")
            .expect("fixture has text_encoder_prior_log_variances");
        let attention_mask = fixture
            .take_tensor("attention_mask")
            .expect("fixture has attention_mask");
        let noise = fixture
            .take_tensor("noise_randn_0")
            .expect("fixture has noise_randn_0 (the duration predictor's noise draw)");
        let ref_log_duration = fixture
            .take_tensor("log_duration")
            .expect("fixture has log_duration");
        let ref_duration = fixture
            .take_tensor("duration")
            .expect("fixture has duration");
        let ref_predicted_lengths = fixture
            .take_tensor("predicted_lengths")
            .expect("fixture has predicted_lengths");
        let ref_attn = fixture.take_tensor("attn").expect("fixture has attn");
        let ref_expanded_means = fixture
            .take_tensor("expanded_prior_means")
            .expect("fixture has expanded_prior_means");
        let ref_expanded_log_vars = fixture
            .take_tensor("expanded_prior_log_variances")
            .expect("fixture has expanded_prior_log_variances");

        let seq_len = attention_mask.shape()[1];
        let batch = attention_mask.shape()[0];

        // HF's `hidden_states = last_hidden_state.transpose(1, 2)` and
        // `input_padding_mask_t = input_padding_mask.transpose(1, 2)`
        // (`input_padding_mask = attention_mask.unsqueeze(-1)`) — a
        // contiguous [batch, seq] reshapes directly to [batch, 1, seq]
        // with identical bytes, since inserting then swapping a size-1
        // axis moves no data.
        let hidden_states_t = last_hidden_state
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .expect("transpose last_hidden_state");
        let padding_mask_t = attention_mask
            .reshape(&[batch, 1, seq_len])
            .expect("reshape attention_mask to [batch, 1, seq]");

        let log_duration = duration_predictor
            .forward_reverse_with_noise(
                &client,
                &hidden_states_t,
                &padding_mask_t,
                None,
                config.noise_scale_duration,
                &noise,
            )
            .unwrap_or_else(|e| panic!("{text}: forward_reverse_with_noise failed: {e}"));
        check_close(
            "log_duration",
            &log_duration,
            &ref_log_duration,
            LOG_DURATION_TOL,
        );

        // Feed the fixture's OWN log_duration (not `log_duration` above) so
        // this isolates regulate_length's arithmetic from unit 1's rounding.
        let regulated = regulate_length(
            &client,
            &ref_log_duration,
            &padding_mask_t,
            &prior_means,
            &prior_log_variances,
            config.speaking_rate,
        )
        .unwrap_or_else(|e| panic!("{text}: regulate_length failed: {e}"));

        check_exact_f32("duration", &regulated.duration, &ref_duration);
        check_exact_i64(
            "predicted_lengths",
            &regulated.predicted_lengths,
            &ref_predicted_lengths,
        );
        check_exact_f32("attn", &regulated.attn, &ref_attn);
        check_close(
            "expanded_prior_means",
            &regulated.expanded_prior_means,
            &ref_expanded_means,
            EXPANDED_PRIOR_TOL,
        );
        check_close(
            "expanded_prior_log_variances",
            &regulated.expanded_prior_log_variances,
            &ref_expanded_log_vars,
            EXPANDED_PRIOR_TOL,
        );
    }
}

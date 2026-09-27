//! Full-pipeline parity for [`VitsModel::synthesize_with_noise`] against
//! HuggingFace's `VitsModel` (`facebook/mms-tts-zlm`).
//!
//! Fixtures come from `tests/tools/vits_reference_dump/vits_reference_dump.py`.
//! NOT checked in (torch/transformers plus the checkpoint), so this skips
//! unless BOTH are present:
//!
//! * the checkpoint — `MMS_TTS_ZLM_DIR`, else
//!   `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`
//! * the dumps — `VITS_REF_DIR`, else `$BOOSTR_MODELS_DIR/vits_reference/mms-tts-zlm`
//!
//! Two checks per sentence:
//!
//! 1. Independently recompute `log_duration` (our text encoder + our
//!    duration predictor, fed the fixture's own `noise_randn_0`) and run it
//!    through `regulate_length` ourselves — this is the one check that
//!    isolates whether OUR log_duration, not the fixture's, still `ceil`s
//!    to the same per-token durations HF's does. `vits_duration_parity.rs`
//!    already pins `log_duration` to ~1e-4 and separately proves
//!    `regulate_length` exact given identical input; this test is the one
//!    that composes them and can catch a `ceil` flip the other two cannot.
//!    A flip is reported by token index and by the pre-ceil log_duration
//!    delta at that token, not papered over with a looser tolerance.
//! 2. Feed the fixture's `input_ids`/`attention_mask` and both captured
//!    noise draws (`noise_randn_0` for the duration predictor,
//!    `noise_randn_like_1` for the prior latents) into
//!    [`VitsModel::synthesize_with_noise`], compare the waveform against
//!    the fixture's `waveform` (signal-to-error floor) and `sequence_lengths`
//!    (exact — it is an integer derived from `predicted_lengths`).

use boostr::model::audio::vits::{
    VitsModel, VitsStochasticDurationPredictor, VitsSynthOptions, VitsTextEncoder, regulate_length,
};
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::{model_fixture, skip_notice};

/// Floor on the waveform's signal-to-error ratio. End to end, the text
/// encoder's f32 rounding passes through the flow and HiFi-GAN, which
/// amplify it, so a per-sample absolute bound tracks the loudest sample
/// rather than the model's correctness. f32 drift lands above 90 dB. A
/// wrong formula (a missing scale, a shifted window, a swapped mask) moves
/// samples by 1e-2 or more, which is below 40 dB.
const WAVEFORM_MIN_SNR_DB: f64 = 80.0;
/// See `vits_duration_parity.rs`'s `LOG_DURATION_TOL` doc.
const LOG_DURATION_TOL: f32 = 1e-4;

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

fn check_snr(name: &str, ours: &Tensor<CpuRuntime>, reference: &Tensor<CpuRuntime>, min_db: f64) {
    assert_eq!(
        ours.shape(),
        reference.shape(),
        "{name}: shape mismatch, ours {:?} reference {:?}",
        ours.shape(),
        reference.shape()
    );
    let a: Vec<f32> = ours.contiguous().unwrap().to_vec();
    let b: Vec<f32> = reference.contiguous().unwrap().to_vec();
    let signal: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    let snr_db = 10.0 * (signal / error).log10();
    let (worst, at) = max_abs_diff(&a, &b);
    eprintln!("{name}: snr {snr_db:.1} dB, max abs diff {worst:e} at flat index {at}");
    assert!(snr_db >= min_db, "{name}: snr {snr_db:.1} dB < {min_db} dB");
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

/// Compares OUR `duration` (`ceil`d from OUR `log_duration`) against the
/// fixture's. Every element must match exactly — `duration` is a `ceil`, so
/// even a 1e-6 pre-ceil drift can flip one when a value sits near an
/// integer boundary. On a mismatch, reports the token index and the
/// pre-ceil `log_duration` delta at that token, rather than loosening the
/// comparison.
fn assert_durations_match_or_report_flip(
    text: &str,
    our_duration: &Tensor<CpuRuntime>,
    ref_duration: &Tensor<CpuRuntime>,
    our_log_duration: &Tensor<CpuRuntime>,
    ref_log_duration: &Tensor<CpuRuntime>,
) {
    assert_eq!(
        our_duration.shape(),
        ref_duration.shape(),
        "{text}: duration shape mismatch, ours {:?} reference {:?}",
        our_duration.shape(),
        ref_duration.shape()
    );
    let ours: Vec<f32> = our_duration.contiguous().unwrap().to_vec();
    let reference: Vec<f32> = ref_duration.contiguous().unwrap().to_vec();
    if ours == reference {
        return;
    }
    let our_ld: Vec<f32> = our_log_duration.contiguous().unwrap().to_vec();
    let ref_ld: Vec<f32> = ref_log_duration.contiguous().unwrap().to_vec();
    for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
        if a != b {
            let our_at = our_ld.get(i).copied().unwrap_or(f32::NAN);
            let ref_at = ref_ld.get(i).copied().unwrap_or(f32::NAN);
            panic!(
                "{text}: duration flipped at token {i}: ours {a} vs reference {b} \
                 (log_duration ours {our_at:e} reference {ref_at:e}, delta {:e})",
                (our_at - ref_at).abs()
            );
        }
    }
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
fn vits_synthesize_with_noise_matches_reference_for_every_sentence() {
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

    // The full model, for the waveform / sequence_lengths check.
    let mut model_weights = VarMap::<CpuRuntime>::from_safetensors(&weights_path, &device)
        .unwrap_or_else(|e| panic!("loading {}: {e}", weights_path.display()));
    let mut model_vb = VarBuilder::new(&mut model_weights, &device);
    let model = VitsModel::<CpuRuntime>::from_varbuilder(&client, &mut model_vb, &config)
        .expect("loading VitsModel from the real checkpoint");

    // A second, independent load of just the text encoder + duration
    // predictor: `VitsModel` does not expose `log_duration` or the
    // pre-regulated `duration` tensor, and this test's whole point is to
    // recompute those from OUR pipeline (see the module doc).
    let mut unit_weights = VarMap::<CpuRuntime>::from_safetensors(&weights_path, &device)
        .unwrap_or_else(|e| panic!("loading {}: {e}", weights_path.display()));
    let mut unit_vb = VarBuilder::new(&mut unit_weights, &device);
    let mut te_vb = unit_vb.pp("text_encoder");
    let text_encoder = VitsTextEncoder::from_varbuilder(&mut te_vb, &config)
        .expect("loading VitsTextEncoder from the real checkpoint");
    drop(te_vb);
    let mut dp_vb = unit_vb.pp("duration_predictor");
    let duration_predictor = VitsStochasticDurationPredictor::from_varbuilder(&mut dp_vb, &config)
        .expect("loading VitsStochasticDurationPredictor from the real checkpoint");
    drop(dp_vb);

    let opts = VitsSynthOptions::from_config(&config);

    for sentence in sentences {
        let file = sentence["file"]
            .as_str()
            .expect("sentence entry has `file`");
        let text = sentence["text"].as_str().unwrap_or("<unknown>");
        let fixture_path = fx.ref_dir.join(file);
        let mut fixture = VarMap::<CpuRuntime>::from_safetensors(&fixture_path, &device)
            .unwrap_or_else(|e| panic!("loading {}: {e}", fixture_path.display()));
        eprintln!("--- {text:?} ---");

        let input_ids = fixture
            .take_tensor("input_ids")
            .expect("fixture has input_ids");
        let attention_mask = fixture
            .take_tensor("attention_mask")
            .expect("fixture has attention_mask");
        let duration_noise = fixture
            .take_tensor("noise_randn_0")
            .expect("fixture has noise_randn_0 (the duration predictor's noise draw)");
        let prior_noise = fixture
            .take_tensor("noise_randn_like_1")
            .expect("fixture has noise_randn_like_1 (the prior-latent noise draw)");
        let ref_waveform = fixture
            .take_tensor("waveform")
            .expect("fixture has waveform");
        let ref_sequence_lengths = fixture
            .take_tensor("sequence_lengths")
            .expect("fixture has sequence_lengths");
        let ref_log_duration = fixture
            .take_tensor("log_duration")
            .expect("fixture has log_duration");
        let ref_duration = fixture
            .take_tensor("duration")
            .expect("fixture has duration");
        let ref_predicted_lengths = fixture
            .take_tensor("predicted_lengths")
            .expect("fixture has predicted_lengths");

        // 1. OUR log_duration -> OUR regulate_length, checked against the
        // fixture's duration / predicted_lengths for an exact match.
        let batch = attention_mask.shape()[0];
        let seq_len = attention_mask.shape()[1];
        let padding_mask = attention_mask
            .reshape(&[batch, seq_len, 1])
            .expect("reshape attention_mask to [batch, seq, 1]");

        let text_out = text_encoder
            .forward(&client, &input_ids, &padding_mask, Some(&attention_mask))
            .unwrap_or_else(|e| panic!("{text}: VitsTextEncoder::forward failed: {e}"));

        let hidden_states = text_out
            .last_hidden_state
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .expect("transpose last_hidden_state");
        let padding_mask_t = padding_mask
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .expect("transpose padding_mask");

        let our_log_duration = duration_predictor
            .forward_reverse_with_noise(
                &client,
                &hidden_states,
                &padding_mask_t,
                None,
                config.noise_scale_duration,
                &duration_noise,
            )
            .unwrap_or_else(|e| panic!("{text}: forward_reverse_with_noise failed: {e}"));
        check_close(
            "log_duration",
            &our_log_duration,
            &ref_log_duration,
            LOG_DURATION_TOL,
        );

        let our_regulated = regulate_length(
            &client,
            &our_log_duration,
            &padding_mask_t,
            &text_out.prior_means,
            &text_out.prior_log_variances,
            config.speaking_rate,
        )
        .unwrap_or_else(|e| panic!("{text}: regulate_length failed: {e}"));

        assert_durations_match_or_report_flip(
            text,
            &our_regulated.duration,
            &ref_duration,
            &our_log_duration,
            &ref_log_duration,
        );
        check_exact_i64(
            "predicted_lengths (from OUR log_duration)",
            &our_regulated.predicted_lengths,
            &ref_predicted_lengths,
        );

        // 2. Full pipeline: waveform + sequence_lengths.
        let out = model
            .synthesize_with_noise(
                &client,
                &input_ids,
                &attention_mask,
                &opts,
                &duration_noise,
                &prior_noise,
            )
            .unwrap_or_else(|e| panic!("{text}: synthesize_with_noise failed: {e}"));

        check_snr(
            "waveform",
            &out.waveform,
            &ref_waveform,
            WAVEFORM_MIN_SNR_DB,
        );
        check_exact_i64(
            "sequence_lengths",
            &out.sequence_lengths,
            &ref_sequence_lengths,
        );
    }
}

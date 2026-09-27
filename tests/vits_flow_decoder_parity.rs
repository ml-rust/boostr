//! Numerical parity for [`sample_prior_latents_with_noise`],
//! [`VitsResidualCouplingBlock`]'s reverse path, and [`VitsHifiGan`]
//! against HuggingFace's `VitsModel` (`facebook/mms-tts-zlm`).
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
//! 1. Prior sampling + flow: feed the fixture's `expanded_prior_means`,
//!    `expanded_prior_log_variances` and its captured
//!    `noise_randn_like_1` draw into
//!    [`sample_prior_latents_with_noise`], compare against the fixture's
//!    `prior_latents`; then run [`VitsResidualCouplingBlock::reverse`] on
//!    our `prior_latents`, compare against the fixture's `flow_output`.
//! 2. Decoder: feed the fixture's OWN `flow_output` (not ours — this
//!    isolates the decoder from step 1's rounding), masked by
//!    `output_padding_mask`, into [`VitsHifiGan::forward`], compare
//!    against the fixture's `waveform`.

use boostr::model::audio::vits::{
    VitsHifiGan, VitsResidualCouplingBlock, sample_prior_latents_with_noise,
};
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::ops::BinaryOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::{model_fixture, skip_notice};

/// See `vits_duration_parity.rs`'s tolerance docs for the reasoning: f32
/// rounding across the flow's conv/WaveNet stack, nothing structural. A
/// wrong flip order, a wrong flow direction, or a dropped mean-only
/// simplification would move this by 1e-2 or more.
const FLOW_TOL: f32 = 1e-4;
/// `sample_prior_latents_with_noise` is one `exp` + two elementwise ops
/// away from the (exact) fixture inputs — f32 accumulation only.
const PRIOR_LATENTS_TOL: f32 = 1e-5;
/// See `vits_text_encoder_parity.rs`'s `TOL` doc: same reasoning, across
/// the HiFi-GAN decoder's much deeper conv/transposed-conv stack.
const WAVEFORM_TOL: f32 = 1e-4;

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
fn vits_flow_and_decoder_match_reference_for_every_sentence() {
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

    let mut flow_vb = vb.pp("flow");
    let flow = VitsResidualCouplingBlock::from_varbuilder(&client, &mut flow_vb, &config)
        .expect("loading VitsResidualCouplingBlock from the real checkpoint");
    drop(flow_vb);

    let mut decoder_vb = vb.pp("decoder");
    let decoder = VitsHifiGan::from_varbuilder(&client, &mut decoder_vb, &config)
        .expect("loading VitsHifiGan from the real checkpoint");
    drop(decoder_vb);

    for sentence in sentences {
        let file = sentence["file"]
            .as_str()
            .expect("sentence entry has `file`");
        let text = sentence["text"].as_str().unwrap_or("<unknown>");
        let fixture_path = fx.ref_dir.join(file);
        let mut fixture = VarMap::<CpuRuntime>::from_safetensors(&fixture_path, &device)
            .unwrap_or_else(|e| panic!("loading {}: {e}", fixture_path.display()));
        eprintln!("--- {text:?} ---");

        let expanded_prior_means = fixture
            .take_tensor("expanded_prior_means")
            .expect("fixture has expanded_prior_means");
        let expanded_prior_log_variances = fixture
            .take_tensor("expanded_prior_log_variances")
            .expect("fixture has expanded_prior_log_variances");
        let prior_noise = fixture
            .take_tensor("noise_randn_like_1")
            .expect("fixture has noise_randn_like_1 (the prior latents' noise draw)");
        let output_padding_mask = fixture
            .take_tensor("output_padding_mask")
            .expect("fixture has output_padding_mask");
        let ref_prior_latents = fixture
            .take_tensor("prior_latents")
            .expect("fixture has prior_latents");
        let ref_flow_output = fixture
            .take_tensor("flow_output")
            .expect("fixture has flow_output");
        let ref_waveform = fixture
            .take_tensor("waveform")
            .expect("fixture has waveform");

        // Step 1: prior sampling.
        let prior_latents = sample_prior_latents_with_noise(
            &client,
            &expanded_prior_means,
            &expanded_prior_log_variances,
            config.noise_scale,
            &prior_noise,
        )
        .unwrap_or_else(|e| panic!("{text}: sample_prior_latents_with_noise failed: {e}"));
        check_close(
            "prior_latents",
            &prior_latents,
            &ref_prior_latents,
            PRIOR_LATENTS_TOL,
        );

        // Step 1: flow reverse, fed OUR prior_latents.
        let flow_output = flow
            .reverse(&client, &prior_latents, &output_padding_mask, None)
            .unwrap_or_else(|e| panic!("{text}: flow reverse failed: {e}"));
        check_close("flow_output", &flow_output, &ref_flow_output, FLOW_TOL);

        // Step 2: decoder, fed the fixture's OWN flow_output (not ours),
        // masked exactly as `VitsModel.forward` does
        // (`spectrogram = latents * output_padding_mask`).
        let spectrogram = client
            .mul(&ref_flow_output, &output_padding_mask)
            .expect("masking flow_output");
        let waveform = decoder
            .forward(&client, &spectrogram, None)
            .unwrap_or_else(|e| panic!("{text}: decoder forward failed: {e}"));
        // HF squeezes the decoder's channel-1 axis (`waveform.squeeze(1)`);
        // ours keeps it, so compare against a reshaped view of the same
        // bytes.
        let batch = ref_waveform.shape()[0];
        let samples = ref_waveform.shape()[1];
        let waveform = waveform
            .reshape(&[batch, samples])
            .expect("reshape waveform to [batch, samples]");
        check_close("waveform", &waveform, &ref_waveform, WAVEFORM_TOL);
    }
}

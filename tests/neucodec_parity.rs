//! Numerical parity for the NeuCodec FSQ quantizer and acoustic decoder
//! against the upstream `neucodec` Python package.
//!
//! The fixtures are produced by running upstream's own `ResidualFSQ`,
//! `VocosBackbone` and `ISTFTHead` on the released weights and dumping raw
//! little-endian f32 blobs. They are NOT checked in (the decoder weights alone
//! are ~560 MB and the checkpoint lives outside the repo), so these tests are
//! skipped unless both of these are present:
//!
//! * `NEUCODEC_REF_DIR` — directory holding `ref_input.f32`, `ref_x_pred.f32`,
//!   `ref_waveform.f32`, `ref_fsq_indices.f32`, `ref_fsq_out.f32`,
//!   `ref_e2e_waveform.f32`
//! * the checkpoint at `NEUCODEC_CHECKPOINT` (defaults to the local path)
//!
//! Regenerate with `dump_reference.py` (see that script's docstring).
//!
//! What this pins that unit tests cannot: the norm family and epsilon inside
//! `ResnetBlock`, the position of the final LayerNorm, the absence of any
//! effective positional encoding, the exp/clamp order in the ISTFT head, and
//! the Vocos `padding="same"` ISTFT trim. Every one of those is invisible in
//! the checkpoint's shapes.

use boostr::model::audio::neucodec::{NeuCodec, NeuCodecDecoder};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::model_fixture;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of f32s",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Largest absolute difference, plus the index where it occurs.
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

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of i32s",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

struct Fixtures {
    dir: PathBuf,
    checkpoint: PathBuf,
}

fn fixtures() -> Option<Fixtures> {
    let dir = PathBuf::from(std::env::var("NEUCODEC_REF_DIR").ok()?);
    let checkpoint = model_fixture("NEUCODEC_CHECKPOINT", "neucodec/model.safetensors")?;
    let needed = [
        "ref_input.f32",
        "ref_x_pred.f32",
        "ref_waveform.f32",
        "ref_fsq_indices.f32",
        "ref_fsq_out.f32",
        "ref_e2e_waveform.f32",
    ];
    if needed.iter().any(|f| !dir.join(f).exists()) {
        return None;
    }
    Some(Fixtures { dir, checkpoint })
}

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn decoder_matches_upstream_neucodec_reference() {
    let Some(fx) = fixtures() else {
        eprintln!(
            "skipping: set NEUCODEC_REF_DIR and (NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR) to run parity"
        );
        return;
    };
    let (client, device) = setup();

    let decoder = NeuCodecDecoder::<CpuRuntime>::from_safetensors(&fx.checkpoint, &device)
        .expect("load decoder");
    let cfg = *decoder.config();

    let input = read_f32(&fx.dir.join("ref_input.f32"));
    let frames = input.len() / cfg.fc_in_dim;
    assert_eq!(
        frames * cfg.fc_in_dim,
        input.len(),
        "ref_input length must be a multiple of fc_in_dim"
    );

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input, &[1, frames, cfg.fc_in_dim], &device).unwrap(),
        false,
    );

    // --- Head inputs: mag/phase vs upstream's pre-split `x_pred` -----------
    //
    // Upstream returns `x_pred = out(x).transpose(1, 2)`, shape [1, 1922, T],
    // i.e. the RAW projection before activation, laid out channels-first. The
    // first 961 rows are log-magnitude, the rest are phase. Applying upstream's
    // own activation (`clamp(exp(m), max=1e2)`) reproduces what our head emits.
    let (mag, phase) = decoder.forward_features(&client, &x).expect("features");
    let f = cfg.n_freq_bins();
    let x_pred = read_f32(&fx.dir.join("ref_x_pred.f32"));
    assert_eq!(x_pred.len(), 2 * f * frames, "unexpected x_pred length");

    let mag_ref: Vec<f32> = x_pred[..f * frames]
        .iter()
        .map(|v| v.exp().min(cfg.mag_clamp_max))
        .collect();
    let phase_ref: Vec<f32> = x_pred[f * frames..].to_vec();

    let mag_got: Vec<f32> = mag.tensor().contiguous().unwrap().to_vec();
    let phase_got: Vec<f32> = phase.tensor().contiguous().unwrap().to_vec();

    let (mag_d, mag_i) = max_abs_diff(&mag_got, &mag_ref);
    let (ph_d, ph_i) = max_abs_diff(&phase_got, &phase_ref);
    eprintln!(
        "mag: max|d|={mag_d:.3e} at {mag_i} (rms {:.3e}); phase: max|d|={ph_d:.3e} at {ph_i} (rms {:.3e})",
        rms(&mag_ref),
        rms(&phase_ref)
    );
    assert!(
        ph_d < 2e-3,
        "phase diverges from upstream: max|d|={ph_d} at index {ph_i}"
    );
    assert!(
        mag_d < 2e-3,
        "magnitude diverges from upstream: max|d|={mag_d} at index {mag_i}"
    );

    // --- End-to-end waveform ---------------------------------------------
    let waveform = decoder.forward(&client, &x).expect("decode");
    let expected_samples = frames * cfg.hop_length;
    assert_eq!(
        waveform.shape(),
        &[1, expected_samples],
        "Vocos same-padding ISTFT must emit exactly hop_length samples per frame"
    );

    let got: Vec<f32> = waveform.contiguous().unwrap().to_vec();
    let want = read_f32(&fx.dir.join("ref_waveform.f32"));
    let (w_d, w_i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    eprintln!("waveform: max|d|={w_d:.3e} at {w_i}, reference rms={scale:.3e}");
    assert!(
        w_d < 1e-3 * scale.max(1.0) + 1e-3,
        "waveform diverges from upstream: max|d|={w_d} at sample {w_i} (rms {scale})"
    );
}

/// The full pure-Rust listening path: FSQ code indices -> 24 kHz waveform,
/// against upstream `ResidualFSQ.get_output_from_indices` + the same decoder.
///
/// Covers what the decoder-only test cannot: the mixed-radix unpack of a
/// 65_536-entry codebook and the `project_out` that follows it. The reference
/// indices deliberately include 0 and 65_535 so the unpack is exercised at both
/// ends of its range.
#[test]
fn codec_matches_upstream_from_indices() {
    let Some(fx) = fixtures() else {
        eprintln!(
            "skipping: set NEUCODEC_REF_DIR and (NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR) to run parity"
        );
        return;
    };
    let (client, device) = setup();

    let codec =
        NeuCodec::<CpuRuntime>::from_safetensors(&fx.checkpoint, &device).expect("load codec");
    let cfg = *codec.config();

    // Indices were dumped as f32 for a uniform fixture format; they are exact
    // small integers, so the round-trip through f32 is lossless.
    let idx_f32 = read_f32(&fx.dir.join("ref_fsq_indices.f32"));
    let frames = idx_f32.len();
    let idx: Vec<i32> = idx_f32.iter().map(|v| *v as i32).collect();
    let indices = Tensor::<CpuRuntime>::from_slice(&idx, &[1, frames], &device).unwrap();

    // --- Dequantization ---------------------------------------------------
    let features = codec
        .indices_to_features(&client, &indices)
        .expect("dequantize");
    assert_eq!(features.shape(), &[1, frames, cfg.fc_in_dim]);

    let got: Vec<f32> = features.tensor().contiguous().unwrap().to_vec();
    let want = read_f32(&fx.dir.join("ref_fsq_out.f32"));
    let (d, i) = max_abs_diff(&got, &want);
    eprintln!(
        "fsq features: max|d|={d:.3e} at {i} (rms {:.3e})",
        rms(&want)
    );
    assert!(
        d < 1e-4,
        "FSQ dequantization diverges from upstream: max|d|={d} at index {i}"
    );

    // --- End to end -------------------------------------------------------
    let waveform = codec.decode(&client, &indices).expect("decode");
    assert_eq!(waveform.shape(), &[1, frames * cfg.hop_length]);

    let got: Vec<f32> = waveform.contiguous().unwrap().to_vec();
    let want = read_f32(&fx.dir.join("ref_e2e_waveform.f32"));
    let (d, i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    eprintln!("e2e waveform: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 1e-3 * scale.max(1.0) + 1e-3,
        "indices->waveform diverges from upstream: max|d|={d} at sample {i} (rms {scale})"
    );
}

/// Reconstruct REAL speech: decode the FSQ codes that upstream's encoder
/// produced for an actual utterance, and compare against upstream's own
/// reconstruction of the same codes.
///
/// The synthetic tests prove the arithmetic matches; this proves it holds over
/// a real 300-frame code sequence (6 s of speech, 293 distinct codes) rather
/// than a handful of hand-picked indices. It also writes the Rust waveform next
/// to the fixtures so it can be converted to WAV and listened to.
///
/// Fixtures come from `encode_real_audio.py`; skipped when absent.
#[test]
fn decodes_real_speech_matching_upstream() {
    let Some(fx) = fixtures() else {
        eprintln!(
            "skipping: set NEUCODEC_REF_DIR and (NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR) to run parity"
        );
        return;
    };
    let idx_path = fx.dir.join("real_indices.i32");
    let ref_path = fx.dir.join("real_ref_waveform.f32");
    if !idx_path.exists() || !ref_path.exists() {
        eprintln!("skipping: real-audio fixtures absent (run encode_real_audio.py)");
        return;
    }
    let (client, device) = setup();

    let codec =
        NeuCodec::<CpuRuntime>::from_safetensors(&fx.checkpoint, &device).expect("load codec");
    let cfg = *codec.config();

    let idx = read_i32(&idx_path);
    let frames = idx.len();
    let indices = Tensor::<CpuRuntime>::from_slice(&idx, &[1, frames], &device).unwrap();

    let waveform = codec.decode(&client, &indices).expect("decode");
    assert_eq!(waveform.shape(), &[1, frames * cfg.hop_length]);

    let got: Vec<f32> = waveform.contiguous().unwrap().to_vec();
    let want = read_f32(&ref_path);

    // Leave the Rust reconstruction beside the fixtures for listening.
    let mut bytes = Vec::with_capacity(got.len() * 4);
    for v in &got {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(fx.dir.join("rust_real_waveform.f32"), &bytes);

    let (d, i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    let err = rms(&got
        .iter()
        .zip(&want)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>());
    eprintln!(
        "real speech: {frames} frames -> {} samples; max|d|={d:.3e} at {i}, \
         rms(err)={err:.3e}, rms(ref)={scale:.3e}, SNR={:.1} dB",
        got.len(),
        20.0 * (scale / err.max(f32::MIN_POSITIVE)).log10()
    );
    assert!(
        d < 1e-3 * scale.max(1.0) + 1e-3,
        "real-speech reconstruction diverges: max|d|={d} at sample {i} (rms {scale})"
    );
}

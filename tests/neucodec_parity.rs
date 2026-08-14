//! Numerical parity for the NeuCodec acoustic decoder against the upstream
//! `neucodec` Python package.
//!
//! The fixtures are produced by running upstream's own `VocosBackbone` +
//! `ISTFTHead` on the released weights and dumping raw little-endian f32
//! blobs. They are NOT checked in (the decoder weights alone are ~560 MB and
//! the checkpoint lives outside the repo), so this test is skipped unless both
//! of these are present:
//!
//! * `NEUCODEC_REF_DIR` — directory holding `ref_input.f32`,
//!   `ref_x_pred.f32`, `ref_waveform.f32`
//! * the checkpoint at `NEUCODEC_CHECKPOINT` (defaults to the local path)
//!
//! Regenerate with `dump_reference.py` (see that script's docstring).
//!
//! What this pins that unit tests cannot: the norm family and epsilon inside
//! `ResnetBlock`, the position of the final LayerNorm, the absence of any
//! effective positional encoding, the exp/clamp order in the ISTFT head, and
//! the Vocos `padding="same"` ISTFT trim. Every one of those is invisible in
//! the checkpoint's shapes.

use boostr::model::audio::neucodec::NeuCodecDecoder;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

const DEFAULT_CHECKPOINT: &str = "/home/farhan/Projects/models/neucodec/model.safetensors";

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of f32s",
        path.display()
    );
    bytes
        .chunks_exact(4)
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

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

struct Fixtures {
    dir: PathBuf,
    checkpoint: PathBuf,
}

fn fixtures() -> Option<Fixtures> {
    let dir = PathBuf::from(std::env::var("NEUCODEC_REF_DIR").ok()?);
    let checkpoint = PathBuf::from(
        std::env::var("NEUCODEC_CHECKPOINT").unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string()),
    );
    let needed = ["ref_input.f32", "ref_x_pred.f32", "ref_waveform.f32"];
    if !checkpoint.exists() || needed.iter().any(|f| !dir.join(f).exists()) {
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
        eprintln!("skipping: set NEUCODEC_REF_DIR (and have the checkpoint) to run parity");
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
        Tensor::<CpuRuntime>::from_slice(&input, &[1, frames, cfg.fc_in_dim], &device),
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

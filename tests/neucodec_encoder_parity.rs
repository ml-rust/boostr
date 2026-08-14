//! Numerical parity for the NeuCodec *encoder* primitives against the upstream
//! `neucodec` Python package.
//!
//! These are the pieces whose exact numerics the checkpoint cannot pin:
//!
//! * the Kaiser-windowed sinc filter taps — upstream registers them as
//!   NON-PERSISTENT buffers, so they are absent from the weights and must be
//!   recomputed bit-comparably here;
//! * `SnakeBeta`, whose `alpha`/`beta` are stored in LOG scale;
//! * the alias-free `Activation1d` (upsample → activation → downsample),
//!   including its replicate padding and asymmetric crops.
//!
//! Fixtures come from `dump_encoder_primitives.py`; skipped when absent.

use boostr::model::audio::neucodec::{
    Activation1d, SnakeBeta, encoder_hop_length, kaiser_sinc_filter1d, load_acoustic_encoder,
};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

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

fn fixtures() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("NEUCODEC_REF_DIR").ok()?);
    let needed = [
        "prim_kaiser_taps.f32",
        "prim_input.f32",
        "prim_alpha.f32",
        "prim_beta.f32",
        "prim_snake_only.f32",
        "prim_upsampled.f32",
        "prim_activation1d.f32",
    ];
    needed.iter().all(|f| dir.join(f).exists()).then_some(dir)
}

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// The filter taps are not in the checkpoint, so a wrong Kaiser `beta` or an
/// off-by-half `time` grid would silently detune every activation in the
/// encoder. Compare against upstream's own `kaiser_sinc_filter1d`.
#[test]
fn kaiser_filter_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let want = read_f32(&dir.join("prim_kaiser_taps.f32"));
    let got = kaiser_sinc_filter1d(0.5 / 2.0, 0.6 / 2.0, 12);
    let (d, i) = max_abs_diff(&got, &want);
    eprintln!("kaiser taps: max|d|={d:.3e} at {i}");
    assert!(
        d < 1e-6,
        "filter taps diverge from upstream: max|d|={d} at {i}"
    );
}

#[test]
fn snake_beta_and_activation1d_match_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let (client, device) = setup();

    let alpha = read_f32(&dir.join("prim_alpha.f32"));
    let beta = read_f32(&dir.join("prim_beta.f32"));
    let input = read_f32(&dir.join("prim_input.f32"));
    let channels = alpha.len();
    let length = input.len() / channels;

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input, &[1, channels, length], &device),
        false,
    );
    let alpha_t = Tensor::<CpuRuntime>::from_slice(&alpha, &[channels], &device);
    let beta_t = Tensor::<CpuRuntime>::from_slice(&beta, &[channels], &device);

    // --- bare SnakeBeta ---------------------------------------------------
    let snake = SnakeBeta::new(alpha_t.clone(), beta_t.clone(), false).unwrap();
    let got: Vec<f32> = snake
        .forward(&client, &x)
        .expect("snake forward")
        .tensor()
        .contiguous()
        .unwrap()
        .to_vec();
    let want = read_f32(&dir.join("prim_snake_only.f32"));
    let (d, i) = max_abs_diff(&got, &want);
    eprintln!("snake_beta: max|d|={d:.3e} at {i}");
    assert!(
        d < 1e-5,
        "SnakeBeta diverges from upstream: max|d|={d} at {i} \
         (a mismatch here usually means alpha/beta were not exponentiated)"
    );

    // --- full alias-free activation ---------------------------------------
    let snake = SnakeBeta::new(alpha_t, beta_t, false).unwrap();
    let act = Activation1d::new(snake, &device).unwrap();

    let up: Vec<f32> = act
        .upsample_for_test(&client, &x)
        .expect("upsample")
        .tensor()
        .contiguous()
        .unwrap()
        .to_vec();
    let want_up = read_f32(&dir.join("prim_upsampled.f32"));
    let (d, i) = max_abs_diff(&up, &want_up);
    eprintln!("upsample: max|d|={d:.3e} at {i}");
    assert!(
        d < 1e-5,
        "UpSample1d diverges from upstream: max|d|={d} at {i}"
    );

    let got: Vec<f32> = act
        .forward(&client, &x)
        .expect("activation1d")
        .tensor()
        .contiguous()
        .unwrap()
        .to_vec();
    let want = read_f32(&dir.join("prim_activation1d.f32"));
    assert_eq!(got.len(), want.len(), "Activation1d must preserve length");
    let (d, i) = max_abs_diff(&got, &want);
    eprintln!("activation1d: max|d|={d:.3e} at {i}");
    assert!(
        d < 1e-5,
        "Activation1d diverges from upstream: max|d|={d} at {i}"
    );
}

/// The full acoustic (BigCodec) encoder against upstream's `CodecEnc`.
///
/// Run on a 3200-sample slice rather than the whole utterance: numr's `conv1d`
/// is a direct convolution, and 6 s through 1536 channels would dominate the
/// suite. The reference encodes the SAME slice, so edge effects match.
///
/// This is the test that pins the whole acoustic stack at once — dilations,
/// the `stride/2 + stride%2` downsample padding, the residual wiring, and the
/// alias-free activations at every stage.
#[test]
fn acoustic_encoder_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let wave_path = dir.join("enc_wave16k_short.f32");
    let ref_path = dir.join("enc_acoustic_short.f32");
    let ckpt = PathBuf::from(
        std::env::var("NEUCODEC_CHECKPOINT")
            .unwrap_or_else(|_| "/home/farhan/Projects/models/neucodec/model.safetensors".into()),
    );
    if !wave_path.exists() || !ref_path.exists() || !ckpt.exists() {
        eprintln!("skipping: acoustic fixtures/checkpoint absent (run encode_real_audio.py)");
        return;
    }
    let (client, device) = setup();

    let encoder =
        load_acoustic_encoder::<CpuRuntime, _>(&ckpt, &device).expect("load acoustic encoder");

    let wave = read_f32(&wave_path);
    let samples = wave.len();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&wave, &[1, 1, samples], &device),
        false,
    );

    let out = encoder.forward(&client, &x).expect("acoustic encode");
    let frames = samples / encoder_hop_length();
    assert_eq!(
        out.shape(),
        &[1, 1024, frames],
        "acoustic encoder must downsample by exactly {}",
        encoder_hop_length()
    );

    let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
    let want = read_f32(&ref_path);
    let (d, i) = max_abs_diff(&got, &want);
    let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("acoustic encoder: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 2e-3 * scale.max(1.0),
        "acoustic encoder diverges from upstream: max|d|={d} at {i} (rms {scale})"
    );
}

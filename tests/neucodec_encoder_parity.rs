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
    Activation1d, NeuCodecEncoder, SnakeBeta, encoder_hop_length, kaiser_sinc_filter1d,
    load_acoustic_encoder, load_residual_fsq, load_semantic_adapter, load_semantic_encoder,
    seamless_fbank,
};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

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

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of i32s",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

/// Path to the real checkpoint, or `None` when it isn't downloaded.
fn checkpoint() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("NEUCODEC_CHECKPOINT")
            .unwrap_or_else(|_| "/home/farhan/Projects/models/neucodec/model.safetensors".into()),
    );
    p.exists().then_some(p)
}

/// The 16-layer Wav2Vec2-BERT semantic encoder, checked at THREE depths so a
/// mismatch localizes instead of just failing:
///   1. feature projection  (LayerNorm-then-Linear, norm over the 160 input dim)
///   2. encoder layer 0 (conformer: ffn1/2 half-residual, relative_key attention,
///      causal depthwise conv module)
///   3. all 16 layers (= upstream `hidden_states[16]`, what NeuCodec reads)
///
/// If (1) passes and (2) fails, the bug is inside the conformer layer; if (2)
/// passes and (3) fails, it is in the stacking/loader.
#[test]
fn semantic_encoder_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let needed = [
        "enc_sem_input.f32",
        "enc_sem_proj.f32",
        "enc_sem_layer0.f32",
        "enc_sem_hidden16.f32",
    ];
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping: checkpoint absent");
        return;
    };
    if needed.iter().any(|f| !dir.join(f).exists()) {
        eprintln!("skipping: semantic fixtures absent (run encode_real_audio.py)");
        return;
    }
    let (client, device) = setup();

    let encoder =
        load_semantic_encoder::<CpuRuntime, _>(&ckpt, &device).expect("load semantic encoder");

    const IN_DIM: usize = 160;
    const HIDDEN: usize = 1024;
    let input = read_f32(&dir.join("enc_sem_input.f32"));
    let frames = input.len() / IN_DIM;
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input, &[1, frames, IN_DIM], &device),
        false,
    );

    let check = |label: &str, got: &[f32], want_file: &str, tol: f32| {
        let want = read_f32(&dir.join(want_file));
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        let (d, i) = max_abs_diff(got, &want);
        let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        eprintln!("{label}: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
        assert!(
            d < tol * scale.max(1.0),
            "{label} diverges from upstream: max|d|={d} at {i} (rms {scale})"
        );
    };

    // 1. feature projection
    let proj = encoder
        .feature_projection()
        .forward(&client, &x)
        .expect("feature projection");
    assert_eq!(proj.shape(), &[1, frames, HIDDEN]);
    let got: Vec<f32> = proj.tensor().contiguous().unwrap().to_vec();
    check("feature projection", &got, "enc_sem_proj.f32", 2e-3);

    // 2. encoder layer 0, fed the projected features
    let l0 = encoder.layers()[0]
        .forward(&client, &proj)
        .expect("encoder layer 0");
    let got: Vec<f32> = l0.tensor().contiguous().unwrap().to_vec();
    check("encoder layer 0", &got, "enc_sem_layer0.f32", 2e-3);

    // 3. the full stack — what NeuCodec actually consumes
    let hs = encoder.forward(&client, &x).expect("semantic encoder");
    assert_eq!(hs.shape(), &[1, frames, HIDDEN]);
    let got: Vec<f32> = hs.tensor().contiguous().unwrap().to_vec();
    check("hidden_states[16]", &got, "enc_sem_hidden16.f32", 3e-3);
}

/// The Kaldi-compatible mel frontend, against upstream's own
/// `SeamlessM4TFeatureExtractor` on a real 1 s waveform.
///
/// This is the single highest-risk piece of the semantic branch: at least nine
/// conventions here have a plausible alternative that yields correctly-shaped
/// but numerically wrong features — Kaldi vs HTK mel scale, triangulating in
/// mel space vs Hz, DC-removal before vs after pre-emphasis, the exact mel
/// floor, `ddof=1` vs population variance, `center=false`, the 2^15 scale, and
/// the Povey window's symmetric-vs-periodic form. Only a numeric comparison
/// against upstream can tell them apart.
#[test]
fn mel_frontend_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let (wave_path, ref_path) = (
        dir.join("enc_fbank_wave.f32"),
        dir.join("enc_fbank_features.f32"),
    );
    if !wave_path.exists() || !ref_path.exists() {
        eprintln!("skipping: fbank fixtures absent (run encode_real_audio.py)");
        return;
    }
    let (_client, device) = setup();

    let wave = read_f32(&wave_path);
    let feats = seamless_fbank::<CpuRuntime>(&wave, &device).expect("fbank");

    const STACKED: usize = 160;
    let want = read_f32(&ref_path);
    let frames = want.len() / STACKED;
    assert_eq!(
        feats.shape(),
        &[frames, STACKED],
        "frame count / stacked width must match upstream"
    );

    let got: Vec<f32> = feats.contiguous().unwrap().to_vec();
    let (d, i) = max_abs_diff(&got, &want);
    let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("mel frontend: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 5e-3,
        "mel frontend diverges from upstream: max|d|={d} at {i} (rms {scale})"
    );
}

/// The semantic adapter (upstream `SemanticEncoder_module`).
///
/// Pins the residual wiring, which is genuinely counter-intuitive: the skip
/// adds `relu(conv1(x))`, not `conv1(x)` and not the raw input, because
/// upstream's first residual-block layer is `nn.ReLU(inplace=True)` and so
/// rewrites the tensor that `residual_blocks(x) + x` goes on to add.
///
/// Every wrong variant still produces correctly-shaped output, which is why
/// this needs a numeric check: the natural `+ conv1(x)` reading is off by
/// `max|d| = 2.10` against an output of rms 1.36.
#[test]
fn semantic_adapter_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let (in_path, ref_path) = (dir.join("enc_sa_input.f32"), dir.join("enc_sa_output.f32"));
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping: checkpoint absent");
        return;
    };
    if !in_path.exists() || !ref_path.exists() {
        eprintln!("skipping: semantic-adapter fixtures absent (run encode_real_audio.py)");
        return;
    }
    let (client, device) = setup();

    let adapter =
        load_semantic_adapter::<CpuRuntime, _>(&ckpt, &device).expect("load semantic adapter");

    let input = read_f32(&in_path);
    const CHANNELS: usize = 1024;
    let frames = input.len() / CHANNELS;
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input, &[1, CHANNELS, frames], &device),
        false,
    );

    let out = adapter.forward(&client, &x).expect("semantic adapter");
    assert_eq!(
        out.shape(),
        &[1, CHANNELS, frames],
        "length must be preserved"
    );

    let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
    let want = read_f32(&ref_path);
    let (d, i) = max_abs_diff(&got, &want);
    let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("semantic adapter: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 2e-3 * scale.max(1.0),
        "semantic adapter diverges from upstream: max|d|={d} at {i} (rms {scale})"
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

/// The `ResidualFSQ` quantizer (`ResidualFsq`) against upstream, on the encode
/// path — the one with the load-bearing double `bound` (see
/// `boostr::nn::fsq::residual` for why collapsing it is wrong).
///
/// Indices are integers: any mismatch is a real divergence, so this reports
/// the mismatch FRACTION and the first few (position, got, want) triples
/// rather than a bare assert — a large fraction (e.g. ~43.75%) means the
/// double bound was lost, while a handful means a float knife-edge in the
/// upstream reference itself.
#[test]
fn residual_fsq_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let (in_path, idx_path, out_path) = (
        dir.join("enc_fsq_input.f32"),
        dir.join("enc_fsq_indices.i32"),
        dir.join("enc_fsq_out.f32"),
    );
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping: checkpoint absent");
        return;
    };
    if !in_path.exists() || !idx_path.exists() || !out_path.exists() {
        eprintln!("skipping: residual FSQ fixtures absent (run encode_real_audio.py)");
        return;
    }
    let (client, device) = setup();

    let quantizer = load_residual_fsq::<CpuRuntime, _>(&ckpt, &device).expect("load residual fsq");

    const DIM: usize = 2048;
    let input = read_f32(&in_path);
    let frames = input.len() / DIM;
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input, &[1, frames, DIM], &device),
        false,
    );

    let (codes, indices) = quantizer.encode(&client, &x).expect("residual fsq encode");

    // `indices` is `[1, frames, num_quantizers]` (num_quantizers = 1 here); the
    // fixture is flat `[1, frames]`. Compare the flattened values, but assert
    // the element count first rather than silently reshaping past a real
    // shape bug.
    let want_indices = read_i32(&idx_path);
    let got_indices: Vec<i32> = indices.contiguous().expect("contiguous indices").to_vec();
    assert_eq!(
        got_indices.len(),
        want_indices.len(),
        "indices element count mismatch (got shape {:?})",
        indices.shape()
    );

    let mismatches: Vec<(usize, i32, i32)> = got_indices
        .iter()
        .zip(want_indices.iter())
        .enumerate()
        .filter_map(|(i, (&g, &w))| (g != w).then_some((i, g, w)))
        .collect();
    if !mismatches.is_empty() {
        let fraction = mismatches.len() as f64 / got_indices.len() as f64 * 100.0;
        let sample: Vec<_> = mismatches.iter().take(5).collect();
        panic!(
            "residual fsq indices diverge from upstream: {}/{} mismatched ({fraction:.2}%); \
             first few (position, got, want): {sample:?}",
            mismatches.len(),
            got_indices.len(),
        );
    }

    let got: Vec<f32> = codes
        .tensor()
        .contiguous()
        .expect("contiguous codes")
        .to_vec();
    let want = read_f32(&out_path);
    assert_eq!(got.len(), want.len(), "quantized_out: length mismatch");
    let (d, i) = max_abs_diff(&got, &want);
    let scale = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("residual fsq quantized_out: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 2e-3 * scale.max(1.0),
        "residual fsq quantized_out diverges from upstream: max|d|={d} at {i} (rms {scale})"
    );
}

/// The FULL encode path — 16 kHz waveform in, FSQ code indices out — against
/// upstream `NeuCodec.encode_code`.
///
/// Two clips, because the interesting failures are at the boundaries:
///
/// * `a` is an EXACT multiple of 320. Upstream's `_prepare_audio` pads
///   `320 - (T % 320)`, which at a multiple appends a full extra 320 samples
///   rather than none. A port that "optimizes" that case away produces one
///   fewer acoustic frame and silently shifts every index.
/// * `b` is not a multiple — the ordinary case.
///
/// Both are checked stage by stage (padding, per-branch frame counts, the
/// post-`fc_prior` prior, then the indices) so a failure says WHERE. The
/// branches deliberately disagree on length — acoustic 26 vs semantic 25 for
/// 8320 samples — and upstream reconciles by truncating both to the minimum,
/// so the frame-count assertions are part of the contract, not incidental.
///
/// The bar is EXACT integer index match. These are discrete codes: a
/// near-miss float at a quantization boundary flips one, so any tolerance
/// would be meaningless. Mismatches are reported as a fraction plus examples.
#[test]
fn full_encode_matches_upstream() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_encoder_primitives.py)");
        return;
    };
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping: checkpoint absent");
        return;
    };
    let clips = ["a", "b"];
    let needed: Vec<PathBuf> = clips
        .iter()
        .flat_map(|c| {
            [
                dir.join(format!("enc_full_{c}_wave.f32")),
                dir.join(format!("enc_full_{c}_padded.f32")),
                dir.join(format!("enc_full_{c}_sem.f32")),
                dir.join(format!("enc_full_{c}_ac.f32")),
                dir.join(format!("enc_full_{c}_prior.f32")),
                dir.join(format!("enc_full_{c}_indices.i32")),
            ]
        })
        .collect();
    if needed.iter().any(|p| !p.exists()) {
        eprintln!("skipping: full-encode fixtures absent (run dump_full_encode.py)");
        return;
    }
    let (client, device) = setup();

    let encoder =
        NeuCodecEncoder::<CpuRuntime>::from_safetensors(&ckpt, &device).expect("load encoder");

    const CHANNELS: usize = 1024;
    const PRIOR: usize = 2048;

    for clip in clips {
        let wave = read_f32(&dir.join(format!("enc_full_{clip}_wave.f32")));
        let stages = encoder
            .encode_stages(&client, &wave, &device)
            .unwrap_or_else(|e| panic!("clip {clip}: encode failed: {e}"));

        // 1. Padding — the always-pad rule, checked against upstream's own
        //    padded waveform rather than recomputed here.
        let want_padded = read_f32(&dir.join(format!("enc_full_{clip}_padded.f32")));
        let got_padded: Vec<f32> = stages
            .padded
            .contiguous()
            .expect("contiguous padded")
            .to_vec();
        assert_eq!(
            got_padded.len(),
            want_padded.len(),
            "clip {clip}: padded length {} != upstream {} \
             (input was {} samples; the pad ALWAYS fires, even at a multiple of 320)",
            got_padded.len(),
            want_padded.len(),
            wave.len(),
        );
        let (d, i) = max_abs_diff(&got_padded, &want_padded);
        assert!(
            d == 0.0,
            "clip {clip}: padded waveform differs at {i} ({d})"
        );

        // 2. Per-branch frame counts, BEFORE truncation. A padding bug shows
        //    up here first, as an off-by-one acoustic frame.
        let want_sem = read_f32(&dir.join(format!("enc_full_{clip}_sem.f32")));
        let want_ac = read_f32(&dir.join(format!("enc_full_{clip}_ac.f32")));
        let ts = stages.semantic.shape()[2];
        let ta = stages.acoustic.shape()[2];
        let min_len = ts.min(ta);
        eprintln!(
            "clip {clip}: padded={} semantic frames={ts} acoustic frames={ta} -> {min_len}",
            got_padded.len(),
        );
        // Upstream's dumps are already truncated to min_len.
        assert_eq!(
            want_sem.len() / CHANNELS,
            min_len,
            "clip {clip}: upstream semantic frames disagree with min(Ts={ts}, Ta={ta})"
        );
        assert_eq!(
            want_ac.len() / CHANNELS,
            min_len,
            "clip {clip}: upstream acoustic frames disagree with min(Ts={ts}, Ta={ta})"
        );

        // 3. The prior — everything after concat + fc_prior, in one number.
        let want_prior = read_f32(&dir.join(format!("enc_full_{clip}_prior.f32")));
        let got_prior: Vec<f32> = stages
            .prior
            .contiguous()
            .expect("contiguous prior")
            .to_vec();
        assert_eq!(
            got_prior.len(),
            want_prior.len(),
            "clip {clip}: prior length mismatch (shape {:?}, expected [1, {PRIOR}, {min_len}])",
            stages.prior.shape(),
        );
        let (d, i) = max_abs_diff(&got_prior, &want_prior);
        let scale =
            (want_prior.iter().map(|v| v * v).sum::<f32>() / want_prior.len() as f32).sqrt();
        eprintln!("clip {clip}: prior max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
        assert!(
            d < 3e-3 * scale.max(1.0),
            "clip {clip}: prior diverges from upstream: max|d|={d} at {i} (rms {scale}). \
             A large, structured error here usually means the concat order was \
             reversed (upstream puts SEMANTIC first)."
        );

        // 4. The indices — exact, no tolerance.
        let want_idx = read_i32(&dir.join(format!("enc_full_{clip}_indices.i32")));
        let got_idx: Vec<i32> = stages
            .indices
            .contiguous()
            .expect("contiguous indices")
            .to_vec();
        assert_eq!(
            got_idx.len(),
            want_idx.len(),
            "clip {clip}: index count mismatch (shape {:?})",
            stages.indices.shape()
        );
        let mismatches: Vec<(usize, i32, i32)> = got_idx
            .iter()
            .zip(want_idx.iter())
            .enumerate()
            .filter_map(|(i, (&g, &w))| (g != w).then_some((i, g, w)))
            .collect();
        let fraction = mismatches.len() as f64 / got_idx.len() as f64 * 100.0;
        eprintln!(
            "clip {clip}: indices {}/{} mismatched ({fraction:.2}%)",
            mismatches.len(),
            got_idx.len()
        );
        assert!(
            mismatches.is_empty(),
            "clip {clip}: FSQ indices diverge from upstream: {}/{} mismatched ({fraction:.2}%); \
             first few (position, got, want): {:?}",
            mismatches.len(),
            got_idx.len(),
            mismatches.iter().take(5).collect::<Vec<_>>(),
        );
    }
}

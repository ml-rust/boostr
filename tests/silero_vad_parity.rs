//! Numerical parity for the Silero VAD model against the upstream ONNX graph.
//!
//! Neither the weights nor the fixture is checked in (they are 1.2 MB of
//! parameters and 60 s of audio), so these tests skip unless
//! `SILERO_VAD_WEIGHTS` / `SILERO_VAD_FIXTURE` point at them, or
//! `BOOSTR_MODELS_DIR` contains `silero-vad/`.
//!
//! Fixture tensors, all f32:
//! * `input` `[960000]` — 60 s of real speech at 16 kHz
//! * `prob` `[1875]` — the ONNX model's speech probability per consecutive
//!   512-sample chunk, from zero state, with the 64-sample context contract
//! * `state_final` `[2, 1, 128]` — `[h, c]` after the last chunk
//!
//! What this pins that unit tests cannot: the 64-sample context carried in
//! FRONT of each chunk, the 64-sample REFLECTION pad after it (upstream uses
//! `nn.ReflectionPad1d`, not zeros), the real/imaginary channel split of the
//! STFT basis convolution, the encoder stride schedule,
//! the PyTorch `[i, f, g, o]` gate order of the LSTM weights, and the ReLU that
//! sits BEFORE the 1x1 output conv. Getting the context wrong is the dangerous
//! one: it does not error, it just returns roughly 0 for every chunk, so the
//! VAD reads as conservative rather than broken.

use boostr::model::audio::SileroVad;
use boostr::nn::VarMap;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

mod common;
use common::{model_fixture, skip_notice};

const CHUNKS: usize = 1875;
const HIDDEN: usize = 128;

/// Max absolute difference allowed against the ONNX reference.
///
/// Every structural way this port can differ from upstream — a dropped
/// context, a missing tail pad, a swapped real/imaginary half, a reordered LSTM
/// gate, a misplaced ReLU — moves the probabilities by 1e-2 or more, so
/// anything above this bound means a formula is wrong, not that the tolerance
/// is tight. Do NOT relax it to paper over a divergence.
const TOL: f32 = 1e-4;

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

fn weights_path() -> Option<std::path::PathBuf> {
    model_fixture(
        "SILERO_VAD_WEIGHTS",
        "silero-vad/silero_vad_16k.safetensors",
    )
}

fn load_model() -> Option<(SileroVad<CpuRuntime>, CpuClient, CpuDevice)> {
    let path = weights_path()?;
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    match SileroVad::<CpuRuntime>::from_safetensors(&path, &device) {
        Ok(vad) => Some((vad, client, device)),
        Err(e) => panic!("failed to load {}: {e}", path.display()),
    }
}

fn load_fixture() -> Option<VarMap<CpuRuntime>> {
    let path = model_fixture(
        "SILERO_VAD_FIXTURE",
        "silero-vad/silero_vad_fixture.safetensors",
    )?;
    let device = CpuDevice::new();
    match VarMap::<CpuRuntime>::from_safetensors(&path, &device) {
        Ok(map) => Some(map),
        Err(e) => panic!("failed to load {}: {e}", path.display()),
    }
}

fn tensor(map: &VarMap<CpuRuntime>, name: &str) -> Vec<f32> {
    map.get_tensor(name)
        .unwrap_or_else(|e| panic!("fixture is missing `{name}`: {e}"))
        .to_vec()
}

#[test]
fn silero_vad_matches_the_onnx_reference() {
    let Some((vad, client, device)) = load_model() else {
        skip_notice("silero vad weights", "SILERO_VAD_WEIGHTS");
        return;
    };
    let Some(map) = load_fixture() else {
        skip_notice("silero vad fixture", "SILERO_VAD_FIXTURE");
        return;
    };

    let input = tensor(&map, "input");
    let reference = tensor(&map, "prob");
    let state_final = tensor(&map, "state_final");
    assert_eq!(
        reference.len(),
        CHUNKS,
        "reference `prob` is not [{CHUNKS}]"
    );
    assert_eq!(
        state_final.len(),
        2 * HIDDEN,
        "reference `state_final` is not [2, 1, {HIDDEN}]"
    );

    let mut state = vad.new_state(&device).expect("fresh VAD state");
    let probs = vad
        .probabilities_with(&client, &mut state, &input)
        .expect("VAD forward");
    assert_eq!(probs.len(), CHUNKS, "wrong chunk count for a 60 s signal");

    let (worst, at) = max_abs_diff(&probs, &reference);
    eprintln!(
        "silero vad prob: max abs diff {worst:e} at chunk {at} \
         (ours {}, reference {})",
        probs[at], reference[at]
    );

    // The recurrent state is where a threading bug hides: per-chunk
    // probabilities can average over a state that drifts slowly, so compare the
    // final (h, c) directly.
    let h = state.hidden().to_vec::<f32>();
    let c = state.cell().to_vec::<f32>();
    let (h_worst, h_at) = max_abs_diff(&h, &state_final[..HIDDEN]);
    eprintln!(
        "silero vad h_final: max abs diff {h_worst:e} at unit {h_at} \
         (ours {}, reference {})",
        h[h_at], state_final[h_at]
    );
    let (c_worst, c_at) = max_abs_diff(&c, &state_final[HIDDEN..]);
    eprintln!(
        "silero vad c_final: max abs diff {c_worst:e} at unit {c_at} \
         (ours {}, reference {})",
        c[c_at],
        state_final[HIDDEN + c_at]
    );

    assert!(worst < TOL, "prob: max abs diff {worst:e} >= {TOL:e}");
    assert!(
        h_worst < TOL,
        "h_final: max abs diff {h_worst:e} >= {TOL:e}"
    );
    assert!(
        c_worst < TOL,
        "c_final: max abs diff {c_worst:e} >= {TOL:e}"
    );
}

#[test]
fn a_chunk_that_is_not_512_samples_is_rejected() {
    let Some((vad, client, device)) = load_model() else {
        skip_notice("silero vad weights", "SILERO_VAD_WEIGHTS");
        return;
    };
    let mut state = vad.new_state(&device).expect("fresh VAD state");
    // Padding a short chunk would return a probability the upstream model never
    // produces, so the model refuses instead.
    let short = vec![0.1f32; 480];
    let long = vec![0.1f32; 1024];
    let exact = vec![0.0f32; 512];
    assert!(
        vad.chunk_probability(&client, &mut state, &short).is_err(),
        "a 480-sample chunk must be rejected"
    );
    assert!(
        vad.chunk_probability(&client, &mut state, &long).is_err(),
        "a 1024-sample chunk must be rejected"
    );
    // The valid length still works, and the rejected calls left no residue.
    let p = vad
        .chunk_probability(&client, &mut state, &exact)
        .expect("512 samples is the valid chunk length");
    assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
}

/// The 64-sample context is the whole reason this model takes 576 samples and
/// not 512, and dropping it does NOT error — it quietly returns a near-zero
/// probability for every chunk, so a broken port reads as a cautious one.
///
/// The context is only ONE of four STFT frames, so a quiet context is
/// indistinguishable from zeros: upstream returns bit-identical `0.094242126`
/// for chunk 500 whether the context is zeroed or the real preceding audio.
/// A loud context is what separates them — upstream gives `0.041625053` there.
/// Both numbers were read off the ONNX model directly.
///
/// This needs the real checkpoint: with synthetic weights every path agrees to
/// the last f32 bit, which says nothing.
#[test]
fn the_context_actually_reaches_the_network() {
    let Some((vad, client, device)) = load_model() else {
        skip_notice("silero vad weights", "SILERO_VAD_WEIGHTS");
        return;
    };
    let Some(map) = load_fixture() else {
        skip_notice("silero vad fixture", "SILERO_VAD_FIXTURE");
        return;
    };
    let input = tensor(&map, "input");
    let chunk_samples = vad.config().chunk_samples;
    let chunk = &input[500 * chunk_samples..501 * chunk_samples];

    let mut fresh = vad.new_state(&device).expect("state");
    let zero_context = vad
        .chunk_probability(&client, &mut fresh, chunk)
        .expect("zero context");

    let mut primed = vad.new_state(&device).expect("state");
    primed
        .set_context(&vec![0.9f32; vad.config().context_samples])
        .expect("prime context");
    let loud_context = vad
        .chunk_probability(&client, &mut primed, chunk)
        .expect("loud context");

    eprintln!("silero vad context: zero {zero_context} vs loud {loud_context}");
    assert!(
        (zero_context - 0.094_242_13).abs() < TOL,
        "zero-context probability drifted from upstream: {zero_context}"
    );
    assert!(
        (loud_context - 0.041_625_05).abs() < TOL,
        "loud-context probability drifted from upstream: {loud_context}"
    );
}

/// A context of the wrong length shifts every STFT frame, so it is refused
/// rather than padded.
#[test]
fn a_context_of_the_wrong_length_is_rejected() {
    let Some((vad, _client, device)) = load_model() else {
        skip_notice("silero vad weights", "SILERO_VAD_WEIGHTS");
        return;
    };
    let mut state = vad.new_state(&device).expect("state");
    assert!(state.set_context(&[0.0; 63]).is_err(), "63 samples");
    assert!(state.set_context(&[0.0; 65]).is_err(), "65 samples");
    assert!(
        state
            .set_context(&vec![0.0; vad.config().context_samples])
            .is_ok()
    );
}

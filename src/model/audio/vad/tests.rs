//! Unit tests for the parts of Silero VAD that do not need the checkpoint:
//! the input contract (chunk length, carried context) and the config geometry.
//!
//! Numerical parity against the upstream ONNX model lives in
//! `tests/silero_vad_parity.rs`, which needs the real weights.

use super::*;
use crate::error::Error;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn patterned(shape: &[usize], scale: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| scale * ((i % 17) as f32 - 8.0)).collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("patterned tensor")
}

/// A structurally correct 16 kHz model with synthetic weights. Every shape
/// matches the real checkpoint, so the whole forward pass runs; only the
/// numbers are meaningless.
fn model(device: &CpuDevice) -> SileroVad<CpuRuntime> {
    let config = VadConfig::silero_16k();
    let bins = config.freq_bins();
    let encoder: Vec<_> = config
        .encoder_channels()
        .iter()
        .map(|&(in_c, out_c)| {
            (
                patterned(&[out_c, in_c, ENCODER_KERNEL], 0.001, device),
                patterned(&[out_c], 0.01, device),
            )
        })
        .collect();
    let weights = SileroVadWeights {
        stft_basis: patterned(&[2 * bins, 1, config.n_fft], 0.002, device),
        encoder,
        rnn_weight_ih: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, device),
        rnn_weight_hh: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, device),
        rnn_bias_ih: patterned(&[4 * HIDDEN_SIZE], 0.01, device),
        rnn_bias_hh: patterned(&[4 * HIDDEN_SIZE], 0.01, device),
        head_weight: patterned(&[1, HIDDEN_SIZE, 1], 0.01, device),
        head_bias: patterned(&[1], 0.02, device),
    };
    SileroVad::new(config, weights).expect("synthetic weights are shape-correct")
}

/// A deterministic, chunk-length signal that is different from every other
/// chunk `seed` produces.
fn chunk(seed: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (seed + i as f32 * 0.01).sin() * 0.5)
        .collect()
}

#[test]
fn sixteen_khz_geometry_matches_the_onnx_graph() {
    let config = VadConfig::silero_16k();
    assert_eq!(config.chunk_samples, 512);
    assert_eq!(config.context_samples, 64);
    assert_eq!(config.freq_bins(), 129);
    assert_eq!(config.hop(), 128);
    // 64 context + 512 chunk + a 64-sample reflection pad.
    assert_eq!(config.window_samples(), 640);
    // The STFT convolution must produce exactly 4 frames.
    assert_eq!(
        (config.window_samples() - config.n_fft) / config.hop() + 1,
        STFT_FRAMES
    );
    assert_eq!(
        config.encoder_channels(),
        [(129, 128), (128, 64), (64, 64), (64, 128)]
    );
}

#[test]
fn eight_khz_geometry_matches_its_checkpoint() {
    let config = VadConfig::silero_8k();
    assert_eq!(config.freq_bins(), 65);
    assert_eq!(config.window_samples(), 320);
    assert_eq!(
        (config.window_samples() - config.n_fft) / config.hop() + 1,
        STFT_FRAMES
    );
    // The 8 kHz first encoder conv is [128, 65, 3] in the checkpoint.
    assert_eq!(config.encoder_channels()[0], (65, 128));
}

#[test]
fn fresh_state_is_all_zeros() {
    let (_client, device) = cpu_setup();
    let config = VadConfig::silero_16k();
    let state = VadState::<CpuRuntime>::new(&config, &device).expect("state");
    assert_eq!(state.context().len(), config.context_samples);
    assert!(state.context().iter().all(|&v| v == 0.0));
    assert_eq!(state.hidden().shape(), &[1, HIDDEN_SIZE]);
    assert_eq!(state.cell().shape(), &[1, HIDDEN_SIZE]);
    assert!(state.hidden().to_vec::<f32>().iter().all(|&v| v == 0.0));
    assert!(state.cell().to_vec::<f32>().iter().all(|&v| v == 0.0));
}

#[test]
fn context_carries_the_previous_chunks_tail() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let mut state = vad.new_state(&device).expect("state");
    let ctx = vad.config().context_samples;
    let len = vad.config().chunk_samples;

    // Chunk 0 sees a zero context.
    assert!(state.context().iter().all(|&v| v == 0.0));

    let first = chunk(0.0, len);
    vad.chunk_probability(&client, &mut state, &first)
        .expect("chunk 0");
    assert_eq!(state.context(), &first[len - ctx..]);

    let second = chunk(3.0, len);
    vad.chunk_probability(&client, &mut state, &second)
        .expect("chunk 1");
    // The tail of chunk 1, NOT of the padded window and not of chunk 0.
    assert_eq!(state.context(), &second[len - ctx..]);
}

#[test]
fn short_chunk_is_an_error_not_a_silent_pad() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let mut state = vad.new_state(&device).expect("state");
    let err = vad
        .chunk_probability(&client, &mut state, &chunk(0.0, 511))
        .expect_err("511 samples must be rejected");
    assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
}

#[test]
fn long_chunk_is_an_error() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let mut state = vad.new_state(&device).expect("state");
    let err = vad
        .chunk_probability(&client, &mut state, &chunk(0.0, 513))
        .expect_err("513 samples must be rejected");
    assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
}

#[test]
fn empty_chunk_is_an_error() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let mut state = vad.new_state(&device).expect("state");
    let err = vad
        .chunk_probability(&client, &mut state, &[])
        .expect_err("an empty chunk must be rejected");
    assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
}

#[test]
fn probabilities_drops_the_trailing_partial_chunk() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let len = vad.config().chunk_samples;
    let samples = chunk(0.0, 3 * len + 100);
    let probs = vad.probabilities(&client, &samples).expect("probabilities");
    assert_eq!(probs.len(), 3);
    assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn probabilities_with_continues_an_existing_stream() {
    // Feeding one block of two chunks must equal feeding two blocks of one,
    // which is only true if the state (h, c AND context) survives the call.
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let len = vad.config().chunk_samples;
    let mut samples = chunk(0.0, len);
    samples.extend(chunk(7.0, len));

    let one_shot = vad.probabilities(&client, &samples).expect("one shot");

    let mut state = vad.new_state(&device).expect("state");
    let mut split = vad
        .probabilities_with(&client, &mut state, &samples[..len])
        .expect("first block");
    split.extend(
        vad.probabilities_with(&client, &mut state, &samples[len..])
            .expect("second block"),
    );

    assert_eq!(one_shot.len(), 2);
    assert_eq!(split.len(), 2);
    for (a, b) in one_shot.iter().zip(split.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn wrong_sample_rate_weights_are_rejected() {
    let (_client, device) = cpu_setup();
    // 8 kHz-shaped first encoder conv against the 16 kHz config.
    let config = VadConfig::silero_16k();
    let bins = config.freq_bins();
    let encoder: Vec<_> = VadConfig::silero_8k()
        .encoder_channels()
        .iter()
        .map(|&(in_c, out_c)| {
            (
                patterned(&[out_c, in_c, ENCODER_KERNEL], 0.001, &device),
                patterned(&[out_c], 0.01, &device),
            )
        })
        .collect();
    let weights = SileroVadWeights {
        stft_basis: patterned(&[2 * bins, 1, config.n_fft], 0.002, &device),
        encoder,
        rnn_weight_ih: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, &device),
        rnn_weight_hh: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, &device),
        rnn_bias_ih: patterned(&[4 * HIDDEN_SIZE], 0.01, &device),
        rnn_bias_hh: patterned(&[4 * HIDDEN_SIZE], 0.01, &device),
        head_weight: patterned(&[1, HIDDEN_SIZE, 1], 0.01, &device),
        head_bias: patterned(&[1], 0.02, &device),
    };
    // `SileroVad` holds tensors and is not `Debug`, so `expect_err` is unusable.
    let Err(err) = SileroVad::<CpuRuntime>::new(config, weights) else {
        panic!("8 kHz encoder must not load as 16 kHz");
    };
    assert!(matches!(err, Error::ModelError { .. }));
}

#[test]
fn a_state_from_the_other_sample_rate_is_rejected() {
    let (client, device) = cpu_setup();
    let vad = model(&device);
    let mut state = VadState::<CpuRuntime>::new(&VadConfig::silero_8k(), &device).expect("state");
    let err = vad
        .chunk_probability(&client, &mut state, &chunk(0.0, 512))
        .expect_err("a 32-sample context must be rejected by the 16 kHz model");
    assert!(matches!(err, Error::InvalidArgument { arg: "state", .. }));
}

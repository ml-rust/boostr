//! **⚠️ Speculative scaffolding — do not wire into production path.**
//!
//! This `KokoroModel` was written from a StyleTTS2 sketch before upstream
//! source inspection. It uses the wrong building blocks in several places
//! (old `DurationPredictor`, `FramePredictor`, `DecoderBlock`, separate
//! `mag_head` + `phase_head` with softplus on magnitude). The real Kokoro
//! forward assembly uses:
//!
//! * `ProsodyPredictor` (M7 prosody module) — replaces both
//!   `DurationPredictor` and `FramePredictor` speculation.
//! * `AdaINResBlock1` (3-tier, Snake-activated) + `AdainResBlk1d`
//!   (single-path, LeakyReLU) — replaces `DecoderBlock`.
//! * `MagPhaseHead` — single `conv_post` + `exp`/`sin` split, replaces the
//!   two-head softplus variant here.
//!
//! The final assembly lives in a new struct wired by `load_kokoro` once the
//! ALBERT backbone port lands. Leave this struct alone until then — patching
//! it piecewise produces a Frankenstein that nothing else uses.
//!
//! Full Kokoro TTS model assembly (OLD scaffolding — see warning above).
//!
//! Holds every submodule and composes them into an end-to-end
//! phonemes-to-waveform pipeline. Retained for reference; `KokoroModel` is
//! not re-exported from the kokoro module's root `use` list and is scheduled
//! for removal once the real assembly is in place.
//!
//! Pipeline:
//!
//! 1. Text encoder: `phoneme_ids [1, T_phon]` → `[1, T_phon, d_model]`
//! 2. Duration predictor: `(hidden, style)` → log-durations `[1, T_phon]`
//! 3. Decode durations (round + clamp) → `Vec<u32>` of length `T_phon`
//! 4. Length regulator: repeat each phoneme's hidden by its duration →
//!    `[1, T_frames, d_model]`
//! 5. Pitch & energy predictors: `(frames, style)` → `[1, T_frames]` each;
//!    broadcast-concat back onto frames
//! 6. Decoder stack: AdaIN-conditioned residual Conv1d blocks interleaved
//!    with transposed-conv upsamples → `[1, C_head, T_spec]`
//! 7. Magnitude and phase heads: two final Conv1d → `(mag, phase) [1, F, T]`
//!    (mag activation via softplus, phase stays linear)
//! 8. iSTFT → waveform `[1, N_samples]`

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{
    DecoderBlock, DurationPredictor, FramePredictor, IStftOptions, StyleProjector, TextEncoder,
    UpsampleBlock, decode_durations, istft, length_regulator,
};
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ComplexOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Style projectors for a single DecoderBlock — two AdaIN sites, each with
/// its own `(gamma, beta)` projector from the voice style vector.
pub struct DecoderBlockStyle<R: Runtime> {
    pub adain1: StyleProjector<R>,
    pub adain2: StyleProjector<R>,
}

/// One stage of the decoder: an UpsampleBlock followed by a stack of
/// DecoderBlocks, each with its own pair of style projectors.
pub struct DecoderStage<R: Runtime> {
    pub upsample: UpsampleBlock<R>,
    pub residuals: Vec<(DecoderBlock<R>, DecoderBlockStyle<R>)>,
}

/// Full Kokoro model.
pub struct KokoroModel<R: Runtime> {
    pub text_encoder: TextEncoder<R>,
    pub duration_predictor: DurationPredictor<R>,
    pub pitch_predictor: FramePredictor<R>,
    pub energy_predictor: FramePredictor<R>,

    /// Projects duration-expanded hidden states into the decoder's initial
    /// channel width (may equal `d_model` or be different depending on
    /// checkpoint).
    pub decoder_input_conv: Conv1d<R>,
    pub decoder_stages: Vec<DecoderStage<R>>,

    /// Final heads: magnitude (positive via softplus) and phase (raw).
    /// Each is a Conv1d `[C_last, F, k]`.
    pub mag_head: Conv1d<R>,
    pub phase_head: Conv1d<R>,

    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub min_frames_per_phoneme: u32,
}

/// Synthesize options threaded through `forward`.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisControls {
    /// Multiplier applied to predicted durations before rounding — the TTS
    /// "speed" knob (>1 speaks faster, <1 slower).
    pub speed: f32,
}

impl Default for SynthesisControls {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

impl<R: Runtime> KokoroModel<R> {
    /// End-to-end synthesis for batch size 1.
    ///
    /// Returns the waveform `[1, N_samples]`. Batch size > 1 is unsupported
    /// (the length regulator requires uniform frame counts; Kokoro TTS runs
    /// one utterance at a time).
    pub fn forward<C>(
        &self,
        client: &C,
        phoneme_ids: &Tensor<R>,
        style: &Tensor<R>,
        controls: SynthesisControls,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + UtilityOps<R>
            + ComplexOps<R>,
        R::Client: IndexingOps<R>,
    {
        // This generic forward covers everything except iSTFT (CPU-only).
        // A CpuRuntime-specialized wrapper below tacks on the waveform step.
        let _ = (client, phoneme_ids, style, controls);
        Err(Error::ModelError {
            reason: "KokoroModel::forward is iSTFT-dependent; use forward_cpu for now".into(),
        })
    }
}

impl KokoroModel<CpuRuntime> {
    /// CPU-specialized end-to-end synthesis returning `[1, N_samples]` f32.
    pub fn forward_cpu(
        &self,
        client: &CpuClient,
        phoneme_ids: &Tensor<CpuRuntime>,
        style: &Tensor<CpuRuntime>,
        controls: SynthesisControls,
    ) -> Result<Tensor<CpuRuntime>> {
        if phoneme_ids.shape().first() != Some(&1) {
            return Err(Error::InvalidArgument {
                arg: "phoneme_ids",
                reason: "Kokoro synthesis requires batch size 1".into(),
            });
        }

        // 1. Text encode: [1, T_phon] -> [1, T_phon, d_model]
        let hidden = self.text_encoder.forward(client, phoneme_ids)?;

        // 2. Predict durations: [1, T_phon]
        let log_dur = self.duration_predictor.forward(client, &hidden, style)?;
        let log_dur_vec: Vec<f32> = log_dur.contiguous().to_vec();
        let speed_ln = controls.speed.ln();
        let adjusted: Vec<f32> = log_dur_vec.iter().map(|&v| v - speed_ln).collect();
        let durations = decode_durations(&adjusted, self.min_frames_per_phoneme);
        if durations.iter().sum::<u32>() == 0 {
            return Err(Error::ModelError {
                reason: "predicted total duration is zero".into(),
            });
        }

        // 3. Expand to frame rate: [1, T_frames, d_model]
        let frames = length_regulator(client, &hidden, &durations)?;

        // 4. Pitch + energy: [1, T_frames] each
        let pitch = self.pitch_predictor.forward(client, &frames, style)?;
        let energy = self.energy_predictor.forward(client, &frames, style)?;

        // 5. Broadcast pitch & energy onto frames along the channel axis.
        // frames [1, T, D]; pitch/energy [1, T] → [1, T, 1] each → concat along dim 2.
        let t_frames = frames.shape()[1];
        let pitch_bt1 = pitch.reshape(&[1, t_frames, 1]).map_err(Error::Numr)?;
        let energy_bt1 = energy.reshape(&[1, t_frames, 1]).map_err(Error::Numr)?;
        let frames_plus = client
            .cat(&[&frames, &pitch_bt1, &energy_bt1], 2)
            .map_err(Error::Numr)?;

        // Transpose to [1, D+2, T] for Conv1d.
        let frames_bct = frames_plus
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous();

        // 6. Initial decoder conv: [1, C_dec, T].
        let mut x = self
            .decoder_input_conv
            .forward_inference(client, &frames_bct)?;

        // 7. Decoder stages: upsample + residual stack (with AdaIN style
        // projections per residual).
        for stage in &self.decoder_stages {
            x = stage.upsample.forward(client, &x)?;
            for (block, block_style) in &stage.residuals {
                let (g1, b1) = block_style.adain1.forward(client, style)?;
                let (g2, b2) = block_style.adain2.forward(client, style)?;
                x = block.forward(client, &x, &g1, &b1, &g2, &b2)?;
            }
        }

        // 8. Magnitude + phase heads. Magnitude goes through softplus so it
        // stays positive (required for iSTFT's mag * e^{iφ} reconstruction).
        let mag_raw = self.mag_head.forward_inference(client, &x)?;
        let mag = client.softplus(&mag_raw).map_err(Error::Numr)?;
        let phase = self.phase_head.forward_inference(client, &x)?;

        // 9. iSTFT. Use a Hann window of length n_fft.
        let window = hann_window(self.n_fft, phoneme_ids.device());
        let opts = IStftOptions {
            hop_length: self.hop_length,
            center: true,
            eps: 1e-8,
        };
        istft(client, &mag, &phase, &window, opts)
    }
}

/// Hann window of length `n` on the given device.
pub fn hann_window(n: usize, device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
    use std::f32::consts::PI;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let ratio = i as f32 / n.max(1) as f32;
            0.5 - 0.5 * (2.0 * PI * ratio).cos()
        })
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[n], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let (_client, device) = cpu_setup();
        let w = hann_window(8, &device);
        let v: Vec<f32> = w.to_vec();
        assert!(v[0].abs() < 1e-6);
        // Hann is symmetric around the midpoint; the mid value peaks near 1.
        assert!(v[4] > 0.9);
    }

    #[test]
    fn synthesis_controls_default_is_unity_speed() {
        let c = SynthesisControls::default();
        assert_eq!(c.speed, 1.0);
    }
}

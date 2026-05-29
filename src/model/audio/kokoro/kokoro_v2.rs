//! `KokoroModelV2` — the real top-level assembly.
//!
//! Supersedes the speculative `KokoroModel` in `model.rs`. Every submodule is
//! a faithful port of the upstream `hexgrad/kokoro` architecture:
//!
//! ```text
//! KokoroModelV2
//! ├── bert             : BertEncoder (AlbertModel + Linear 768→512)
//! ├── text_encoder     : TextEncoder (embed + Conv1d×3 + BiLSTM)
//! ├── predictor        : ProsodyPredictor (duration/F0/N heads)
//! ├── decoder          : Decoder (asr_res + encode + decode×4 + Generator)
//! └── config           : KokoroConfig
//! ```
//!
//! Forward flow (from upstream `forward_with_tokens`):
//!
//! ```text
//! d_en       = bert(ids).transpose                       [B, 512, T_phon]
//! (dec_s, pred_s) = split_voice_style(ref_s, 128)
//! d          = predictor.text_encoder(d_en, pred_s)      [B, T_phon, 512+128]
//! dur_logits = predictor.duration_proj(predictor.lstm(d))[B, T_phon, 50]
//! durations  = decode_prosody_durations(...)             Vec<u32>
//! en         = length_regulator(d, durations)            [B, T_frames, d_model]
//! (f0, n)    = predictor.predict_f0_n(en, pred_s)        [B, T_frames] × 2
//! t_en       = text_encoder(ids)                         [B, T_phon, 512]
//! asr        = t_en @ alignment(durations)               [B, 512, T_frames]
//! (mag, phase) = decoder(asr, f0, n, dec_s)              [B, F, T_spec] × 2
//! waveform   = istft(mag, phase, window, hop)            [B, N_samples]
//! ```
//!
//! The **alignment** step is what pairs a per-phoneme `text_encoder` output
//! with the duration-expanded frame timeline. Upstream builds an explicit
//! alignment matrix from the decoded `durations`, then multiplies
//! `t_en @ aln_trg`. We reproduce that as [`alignment_matrix_from_durations`]
//! below — it's deterministic once durations are known.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{
    BertEncoder, Decoder, IStftOptions, KokoroConfig, ProsodyPredictor, TextEncoder,
    decode_prosody_durations, istft, length_regulator, split_voice_style,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct KokoroModelV2<R: Runtime> {
    pub bert: BertEncoder<R>,
    pub text_encoder: TextEncoder<R>,
    pub predictor: ProsodyPredictor<R>,
    pub decoder: Decoder<R>,
    pub config: KokoroConfig,
}

impl<R: Runtime> KokoroModelV2<R> {
    /// Generic forward to the (mag, phase) stage. Does not include iSTFT
    /// (which is CPU-only for now because overlap-add needs `scatter_add`).
    /// Use [`KokoroModelV2::synthesize_cpu`] for end-to-end synthesis on
    /// `CpuRuntime`.
    #[allow(clippy::type_complexity)]
    pub fn forward_to_spectrogram<C>(
        &self,
        client: &C,
        token_ids: &Tensor<R>,
        voice_row: &Tensor<R>,
        min_frames_per_phoneme: u32,
    ) -> Result<(Tensor<R>, Tensor<R>, Vec<u32>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + CompareOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        // 1. Split voice row into decoder + predictor halves.
        let (decoder_style, predictor_style) = split_voice_style(voice_row, self.config.style_dim)?;

        // 2. BERT + bert_encoder projection.
        let d_en = self.bert.forward(client, token_ids)?; // [B, T_phon, 512]

        // 3. Duration prediction.
        let dur_logits = self
            .predictor
            .predict_duration(client, &d_en, &predictor_style)?;
        let logits_shape = dur_logits.shape();
        let (b, t_phon) = (logits_shape[0], logits_shape[1]);
        if b != 1 {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: "synthesis is single-utterance; batch > 1 not supported".into(),
            });
        }
        let logits_flat: Vec<f32> = dur_logits.contiguous()?.to_vec();
        let durations = decode_prosody_durations(
            &logits_flat,
            t_phon,
            self.config.max_dur,
            min_frames_per_phoneme,
        );

        // 4. Length regulator on BERT features (d_en) → frame-rate hidden.
        let frames = length_regulator(client, &d_en, &durations)?;

        // 5. F0 + N branches.
        let (f0, n_energy) = self
            .predictor
            .predict_f0_n(client, &frames, &predictor_style)?;

        // 6. Standalone text encoder → [B, T_phon, 512].
        let t_en = self.text_encoder.forward(client, token_ids)?;

        // 7. Alignment-expanded ASR features: each phoneme's t_en row
        // repeated `durations[i]` times, then transposed to [B, C, T_frames].
        let asr_bt_d = length_regulator(client, &t_en, &durations)?; // [1, T_frames, 512]
        let asr = asr_bt_d
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;

        // 8. Decoder → (mag, phase).
        let (mag, phase) = self
            .decoder
            .forward(client, &asr, &f0, &n_energy, &decoder_style)?;
        Ok((mag, phase, durations))
    }
}

impl KokoroModelV2<CpuRuntime> {
    /// CPU-specialized spectrogram forward — runs the decoder through its
    /// `forward_cpu_full` path so the generator's noise conditioning is
    /// applied when the checkpoint provided it.
    #[allow(clippy::type_complexity)]
    pub fn forward_to_spectrogram_cpu(
        &self,
        client: &CpuClient,
        token_ids: &Tensor<CpuRuntime>,
        voice_row: &Tensor<CpuRuntime>,
        min_frames_per_phoneme: u32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Vec<u32>)> {
        let (decoder_style, predictor_style) = split_voice_style(voice_row, self.config.style_dim)?;
        let d_en = self.bert.forward(client, token_ids)?;
        let dur_logits = self
            .predictor
            .predict_duration(client, &d_en, &predictor_style)?;
        let logits_shape = dur_logits.shape();
        let (b, t_phon) = (logits_shape[0], logits_shape[1]);
        if b != 1 {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: "synthesis is single-utterance; batch > 1 not supported".into(),
            });
        }
        let logits_flat: Vec<f32> = dur_logits.contiguous()?.to_vec();
        let durations = decode_prosody_durations(
            &logits_flat,
            t_phon,
            self.config.max_dur,
            min_frames_per_phoneme,
        );
        let frames = length_regulator(client, &d_en, &durations)?;
        let (f0, n_energy) = self
            .predictor
            .predict_f0_n(client, &frames, &predictor_style)?;
        let t_en = self.text_encoder.forward(client, token_ids)?;
        let asr_bt_d = length_regulator(client, &t_en, &durations)?;
        let asr = asr_bt_d
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let (mag, phase) =
            self.decoder
                .forward_cpu_full(client, &asr, &f0, &n_energy, &decoder_style)?;
        Ok((mag, phase, durations))
    }

    /// CPU end-to-end synthesis.
    ///
    /// `token_ids [1, T_phon]` i64 + `voice_row [1, 256]` f32 → waveform
    /// `[1, N_samples]` f32.
    pub fn synthesize_cpu(
        &self,
        client: &CpuClient,
        token_ids: &Tensor<CpuRuntime>,
        voice_row: &Tensor<CpuRuntime>,
        min_frames_per_phoneme: u32,
    ) -> Result<Tensor<CpuRuntime>> {
        let (mag, phase, _durations) =
            self.forward_to_spectrogram_cpu(client, token_ids, voice_row, min_frames_per_phoneme)?;
        let window = super::model::hann_window(self.config.n_fft, voice_row.device());
        let opts = IStftOptions {
            hop_length: self.config.hop_length,
            center: true,
            eps: 1e-8,
        };
        istft(client, &mag, &phase, &window, opts)
    }
}

/// Build an alignment matrix `[T_phon, T_frames]` from integer durations.
///
/// Entry `(p, f)` is `1.0` iff frame `f` belongs to phoneme `p`, else `0.0`.
/// Matches upstream `pred_aln_trg` construction. Returned on the same device
/// as the reference tensor passed in.
pub fn alignment_matrix_from_durations<R: Runtime<DType = DType>>(
    durations: &[u32],
    reference: &Tensor<R>,
) -> Result<Tensor<R>> {
    let t_phon = durations.len();
    let t_frames: u32 = durations.iter().sum();
    if t_frames == 0 {
        return Err(Error::InvalidArgument {
            arg: "durations",
            reason: "total frame count must be > 0".into(),
        });
    }
    let t_frames = t_frames as usize;
    let mut data = vec![0.0f32; t_phon * t_frames];
    let mut cursor = 0usize;
    for (p, &d) in durations.iter().enumerate() {
        for f in 0..(d as usize) {
            data[p * t_frames + cursor + f] = 1.0;
        }
        cursor += d as usize;
    }
    Ok(Tensor::<R>::from_slice(
        &data,
        &[t_phon, t_frames],
        reference.device(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn alignment_matrix_shape_and_placement() {
        let device = CpuDevice::new();
        let reference = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
        let aln = alignment_matrix_from_durations(&[2, 1, 3], &reference).unwrap();
        assert_eq!(aln.shape(), &[3, 6]);
        let v: Vec<f32> = aln.to_vec();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,  // phoneme 0: frames 0..2
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  // phoneme 1: frame 2
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0,  // phoneme 2: frames 3..6
        ];
        assert_eq!(v, expected);
    }

    #[test]
    fn alignment_matrix_rejects_zero_total() {
        let device = CpuDevice::new();
        let reference = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
        assert!(alignment_matrix_from_durations(&[0, 0], &reference).is_err());
    }

    #[test]
    fn alignment_matrix_single_phoneme() {
        let device = CpuDevice::new();
        let reference = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
        let aln = alignment_matrix_from_durations(&[4], &reference).unwrap();
        assert_eq!(aln.shape(), &[1, 4]);
        let v: Vec<f32> = aln.to_vec();
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);
    }
}

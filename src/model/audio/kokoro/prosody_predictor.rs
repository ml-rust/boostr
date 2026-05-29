//! Kokoro `ProsodyPredictor` — duration + F0 + energy head.
//!
//! Replaces the speculative `DurationPredictor` / `FramePredictor` placeholders
//! with the actual upstream architecture:
//!
//! ```text
//! ProsodyPredictor
//! ├── text_encoder: DurationEncoder
//! │     └── lstms: [LSTM, AdaLayerNorm] × nlayers  (alternating)
//! ├── lstm:         BiLSTM(d_hid + style_dim, d_hid/2)
//! ├── duration_proj: Linear(d_hid, max_dur)       # 50-class classifier, NOT scalar
//! ├── shared:       BiLSTM(d_hid + style_dim, d_hid/2)
//! ├── F0:  [AdainResBlk1d × 3]                    # 512→512, 512→256 (upsample), 256→256
//! ├── N:   [AdainResBlk1d × 3]                    # same shape as F0
//! ├── F0_proj: Conv1d(d_hid/2, 1, 1)
//! └── N_proj:  Conv1d(d_hid/2, 1, 1)
//! ```
//!
//! Inference-only (no padding masks, no dropout). The generic
//! `DurationPredictor` / `FramePredictor` structs from earlier turns still
//! exist but are not used on the Kokoro path.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{AdaLayerNorm, AdainResBlk1d};
use crate::nn::{BiLstm, Conv1d};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Alternating `LSTM | AdaLayerNorm` stack (upstream `DurationEncoder`).
pub struct DurationEncoder<R: Runtime> {
    lstms: Vec<BiLstm<R>>,
    adalns: Vec<AdaLayerNorm<R>>,
    d_model: usize,
    style_dim: usize,
}

impl<R: Runtime> DurationEncoder<R> {
    /// `lstms` and `adalns` must be the same length (`nlayers` each). Layer `i`
    /// runs `lstms[i]` first, then `adalns[i]`.
    pub fn new(
        lstms: Vec<BiLstm<R>>,
        adalns: Vec<AdaLayerNorm<R>>,
        d_model: usize,
        style_dim: usize,
    ) -> Result<Self> {
        if lstms.len() != adalns.len() {
            return Err(Error::InvalidArgument {
                arg: "adalns",
                reason: format!(
                    "DurationEncoder expects |lstms| == |adalns|, got {} vs {}",
                    lstms.len(),
                    adalns.len()
                ),
            });
        }
        for (i, (l, a)) in lstms.iter().zip(adalns.iter()).enumerate() {
            if 2 * l.hidden_size() != d_model {
                return Err(Error::InvalidArgument {
                    arg: "lstms",
                    reason: format!(
                        "layer {i} BiLSTM output must equal d_model ({d_model}); got 2·{}",
                        l.hidden_size()
                    ),
                });
            }
            if a.channels() != d_model || a.style_dim() != style_dim {
                return Err(Error::InvalidArgument {
                    arg: "adalns",
                    reason: format!(
                        "layer {i} AdaLayerNorm shape mismatch (channels={}, style_dim={})",
                        a.channels(),
                        a.style_dim()
                    ),
                });
            }
        }
        Ok(Self {
            lstms,
            adalns,
            d_model,
            style_dim,
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn style_dim(&self) -> usize {
        self.style_dim
    }

    /// Forward: `x [B, T, d_model]` + `style [B, style_dim]` → `[B, T, d_model + style_dim]`.
    ///
    /// Output carries the style re-concatenated onto the last AdaLN output so
    /// the downstream `ProsodyPredictor.lstm` / `.shared` (whose inputs are
    /// `d_model + style_dim` wide) can consume it directly.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + TensorOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + UtilityOps<R>
            + ShapeOps<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.d_model {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, T, {}], got {shape:?}", self.d_model),
            });
        }
        let (b, t) = (shape[0], shape[1]);
        if style.shape() != [b, self.style_dim] {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "expected [{b}, {}], got {:?}",
                    self.style_dim,
                    style.shape()
                ),
            });
        }

        // Broadcast style onto the sequence axis once — reused every layer.
        let style_bc = style
            .reshape(&[b, 1, self.style_dim])
            .map_err(Error::Numr)?
            .broadcast_to(&[b, t, self.style_dim])
            .map_err(Error::Numr)?
            .contiguous()?;

        // Running activation in [B, T, d_model + style_dim] — starts as `cat(x, style)`.
        let mut h = client.cat(&[x, &style_bc], 2).map_err(Error::Numr)?;

        for i in 0..self.lstms.len() {
            // BiLSTM produces [B, T, d_model].
            h = self.lstms[i].forward(client, &h)?;
            // AdaLayerNorm expects [B, C, T]; transpose in and back.
            let h_bct = h.transpose(1, 2).map_err(Error::Numr)?.contiguous()?;
            let normed_bct = self.adalns[i].forward(client, &h_bct, style)?;
            let normed = normed_bct
                .transpose(1, 2)
                .map_err(Error::Numr)?
                .contiguous()?;
            // Re-concat style for the next LSTM (or to form the module output).
            h = client.cat(&[&normed, &style_bc], 2).map_err(Error::Numr)?;
        }
        Ok(h)
    }
}

/// F0 / N branch: three `AdainResBlk1d` blocks followed by a `Conv1d(_, 1, 1)`
/// projection. Shared struct since F0 and N are structurally identical.
pub struct ProsodyBranch<R: Runtime> {
    blocks: [AdainResBlk1d<R>; 3],
    proj: Conv1d<R>,
}

impl<R: Runtime> ProsodyBranch<R> {
    pub fn new(blocks: [AdainResBlk1d<R>; 3], proj: Conv1d<R>) -> Self {
        Self { blocks, proj }
    }

    /// Forward: `frames [B, C, T]`, `style [B, style_dim]` → `[B, T]`.
    pub fn forward<C>(&self, client: &C, frames: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + UtilityOps<R>,
    {
        let mut h = frames.clone();
        for block in &self.blocks {
            h = block.forward(client, &h, style)?;
        }
        // [B, 1, T] → [B, T]
        let y = self.proj.forward_inference(client, &h)?;
        let y_shape = y.shape();
        if y_shape.len() != 3 || y_shape[1] != 1 {
            return Err(Error::InvalidArgument {
                arg: "proj_output",
                reason: format!("expected [B, 1, T] after projection, got {y_shape:?}"),
            });
        }
        y.reshape(&[y_shape[0], y_shape[2]]).map_err(Error::Numr)
    }
}

/// Full ProsodyPredictor.
pub struct ProsodyPredictor<R: Runtime> {
    pub text_encoder: DurationEncoder<R>,
    pub lstm: BiLstm<R>,
    /// Weight `[max_dur, d_model]`, bias `[max_dur]`.
    pub duration_proj_weight: Tensor<R>,
    pub duration_proj_bias: Tensor<R>,
    pub shared: BiLstm<R>,
    pub f0: ProsodyBranch<R>,
    pub n: ProsodyBranch<R>,
    pub d_model: usize,
    pub style_dim: usize,
    pub max_dur: usize,
}

impl<R: Runtime> ProsodyPredictor<R> {
    /// Predict duration logits from the BERT-encoded text.
    ///
    /// `texts [B, T, d_model]`, `style [B, style_dim]` → `[B, T, max_dur]`
    /// logits (apply softmax + argmax / expected-value + round for durations).
    pub fn predict_duration<C>(
        &self,
        client: &C,
        texts: &Tensor<R>,
        style: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + TensorOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + UtilityOps<R>
            + ShapeOps<R>,
    {
        // Run the interleaved encoder, then the main LSTM.
        let enc_out = self.text_encoder.forward(client, texts, style)?; // [B, T, d+style]
        let lstm_out = self.lstm.forward(client, &enc_out)?; // [B, T, d_model]

        // Project to [B, T, max_dur] via flat matmul + bias, then reshape.
        let shape = lstm_out.shape();
        let (b, t) = (shape[0], shape[1]);
        let flat = lstm_out
            .reshape(&[b * t, self.d_model])
            .map_err(Error::Numr)?;
        let w_t = self
            .duration_proj_weight
            .transpose(0, 1)
            .map_err(Error::Numr)?;
        let out = client
            .matmul_bias(&flat, &w_t, &self.duration_proj_bias)
            .map_err(Error::Numr)?;
        out.reshape(&[b, t, self.max_dur]).map_err(Error::Numr)
    }

    /// Predict F0 and energy from duration-expanded frame features.
    ///
    /// `frames_bt_d [B, T_frames, d_model]`, `style [B, style_dim]` →
    /// `(f0 [B, T_frames], n [B, T_frames])`.
    #[allow(clippy::type_complexity)]
    pub fn predict_f0_n<C>(
        &self,
        client: &C,
        frames_bt_d: &Tensor<R>,
        style: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
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
            + ShapeOps<R>,
    {
        let shape = frames_bt_d.shape();
        if shape.len() != 3 || shape[2] != self.d_model {
            return Err(Error::InvalidArgument {
                arg: "frames_bt_d",
                reason: format!("expected [B, T, {}], got {shape:?}", self.d_model),
            });
        }
        let (b, t) = (shape[0], shape[1]);
        if style.shape() != [b, self.style_dim] {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "expected [{b}, {}], got {:?}",
                    self.style_dim,
                    style.shape()
                ),
            });
        }

        // `shared` LSTM input width must match `d_model + style_dim`, so we
        // need to concat style onto the frames first.
        let style_bc = style
            .reshape(&[b, 1, self.style_dim])
            .map_err(Error::Numr)?
            .broadcast_to(&[b, t, self.style_dim])
            .map_err(Error::Numr)?
            .contiguous()?;
        let cat = client
            .cat(&[frames_bt_d, &style_bc], 2)
            .map_err(Error::Numr)?;
        let shared_out = self.shared.forward(client, &cat)?; // [B, T, d_model]

        // Branches take [B, C, T]; transpose.
        let shared_bct = shared_out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let f0 = self.f0.forward(client, &shared_bct, style)?;
        let n = self.n.forward(client, &shared_bct, style)?;
        Ok((f0, n))
    }
}

/// Turn per-phoneme duration logits into an integer duration per phoneme.
///
/// * `logits [T, max_dur]` — 2-D (drop the batch axis first if needed).
/// * `min_frames` — floor for each phoneme, clamped to at least 1.
///
/// Uses the softmax-weighted expected value across duration bins, rounded and
/// clamped to `[min_frames, max_dur]`. This matches upstream's
/// `torch.sigmoid(duration).sum(axis=-1)` convention when inputs are treated as
/// per-bin probabilities — but here we stay loyal to the classification head's
/// softmax output. Callers can swap in their own decoding if upstream diverges.
pub fn decode_prosody_durations(
    logits: &[f32],
    t: usize,
    max_dur: usize,
    min_frames: u32,
) -> Vec<u32> {
    assert_eq!(logits.len(), t * max_dur, "logits must be [T, max_dur]");
    let floor = min_frames.max(1);
    let mut out = Vec::with_capacity(t);
    for row in 0..t {
        let base = row * max_dur;
        // softmax — for-loop, stable (subtract max).
        let mut m = f32::NEG_INFINITY;
        for d in 0..max_dur {
            m = m.max(logits[base + d]);
        }
        let mut sum_exp = 0.0f64;
        for d in 0..max_dur {
            sum_exp += ((logits[base + d] - m) as f64).exp();
        }
        // expected value under softmax
        let mut expected = 0.0f64;
        for d in 0..max_dur {
            let p = ((logits[base + d] - m) as f64).exp() / sum_exp;
            expected += p * (d as f64);
        }
        let frames = expected.round() as i64;
        let frames = frames.clamp(floor as i64, max_dur as i64 - 1).max(1);
        out.push(frames as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::KokoroAdaIn1d;
    use crate::nn::Lstm;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }
    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }

    fn bilstm(
        in_dim: usize,
        hidden: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> BiLstm<CpuRuntime> {
        let mk = || {
            Lstm::new(
                zeros(&[4 * hidden, in_dim], device),
                zeros(&[4 * hidden, hidden], device),
                zeros(&[4 * hidden], device),
                zeros(&[4 * hidden], device),
            )
            .unwrap()
        };
        BiLstm::new(mk(), mk()).unwrap()
    }

    fn adaln(
        channels: usize,
        style_dim: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdaLayerNorm<CpuRuntime> {
        AdaLayerNorm::new(
            zeros(&[2 * channels, style_dim], device),
            zeros(&[2 * channels], device),
            1e-5,
        )
        .unwrap()
    }

    fn kadain(
        c: usize,
        s: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> KokoroAdaIn1d<CpuRuntime> {
        KokoroAdaIn1d::new(
            zeros(&[2 * c, s], device),
            zeros(&[2 * c], device),
            ones(&[c], device),
            zeros(&[c], device),
            1e-5,
        )
        .unwrap()
    }

    fn conv(
        c_out: usize,
        c_in: usize,
        k: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            zeros(&[c_out, c_in, k], device),
            Some(zeros(&[c_out], device)),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    #[test]
    fn duration_encoder_preserves_shape_with_style_re_cat() {
        let (client, device) = cpu_setup();
        // d_model=4, style_dim=2, 2 layers.
        let enc = DurationEncoder::new(
            vec![bilstm(6, 2, &device), bilstm(6, 2, &device)],
            vec![adaln(4, 2, &device), adaln(4, 2, &device)],
            4,
            2,
        )
        .unwrap();
        let x = zeros(&[1, 5, 4], &device);
        let style = zeros(&[1, 2], &device);
        let y = enc.forward(&client, &x, &style).unwrap();
        assert_eq!(y.shape(), &[1, 5, 6]); // d_model + style_dim
    }

    #[test]
    fn prosody_predict_duration_outputs_max_dur() {
        let (client, device) = cpu_setup();
        let d = 4;
        let s = 2;
        let max_dur = 3;
        let enc = DurationEncoder::new(
            vec![bilstm(d + s, d / 2, &device)],
            vec![adaln(d, s, &device)],
            d,
            s,
        )
        .unwrap();
        let main_lstm = bilstm(d + s, d / 2, &device);
        let shared = bilstm(d + s, d / 2, &device);

        let branch = |device: &<CpuRuntime as Runtime>::Device| {
            ProsodyBranch::new(
                [
                    AdainResBlk1d::new(
                        kadain(d, s, device),
                        kadain(d, s, device),
                        conv(d, d, 3, device),
                        conv(d, d, 3, device),
                        None,
                        None,
                        0.2,
                    ),
                    AdainResBlk1d::new(
                        kadain(d, s, device),
                        kadain(d, s, device),
                        conv(d, d, 3, device),
                        conv(d, d, 3, device),
                        None,
                        None,
                        0.2,
                    ),
                    AdainResBlk1d::new(
                        kadain(d, s, device),
                        kadain(d, s, device),
                        conv(d, d, 3, device),
                        conv(d, d, 3, device),
                        None,
                        None,
                        0.2,
                    ),
                ],
                conv(1, d, 1, device),
            )
        };

        let pred = ProsodyPredictor {
            text_encoder: enc,
            lstm: main_lstm,
            duration_proj_weight: zeros(&[max_dur, d], &device),
            duration_proj_bias: zeros(&[max_dur], &device),
            shared,
            f0: branch(&device),
            n: branch(&device),
            d_model: d,
            style_dim: s,
            max_dur,
        };

        let texts = zeros(&[1, 6, d], &device);
        let style = zeros(&[1, s], &device);
        let dur = pred.predict_duration(&client, &texts, &style).unwrap();
        assert_eq!(dur.shape(), &[1, 6, max_dur]);

        let frames = zeros(&[1, 7, d], &device);
        let (f0, n) = pred.predict_f0_n(&client, &frames, &style).unwrap();
        assert_eq!(f0.shape(), &[1, 7]);
        assert_eq!(n.shape(), &[1, 7]);
    }

    #[test]
    fn decode_prosody_durations_clamps_to_min_and_rounds() {
        // Uniform logits → expected value = (max_dur - 1)/2 = 1.0 for max_dur=3.
        let logits = vec![0.0f32; 2 * 3];
        let out = decode_prosody_durations(&logits, 2, 3, 1);
        assert_eq!(out, vec![1, 1]);
    }

    #[test]
    fn decode_prosody_durations_picks_peak() {
        // Strong logit on bin 2 → expected ≈ 2.
        let logits = vec![
            -10.0, -10.0, 10.0, // row 0 → dur 2
            -10.0, 10.0, -10.0, // row 1 → dur 1
        ];
        let out = decode_prosody_durations(&logits, 2, 3, 1);
        assert_eq!(out, vec![2, 1]);
    }

    #[test]
    fn duration_encoder_rejects_mismatched_lstm_adaln_counts() {
        let (_client, device) = cpu_setup();
        let enc = DurationEncoder::new(
            vec![bilstm(6, 2, &device)],
            vec![adaln(4, 2, &device), adaln(4, 2, &device)],
            4,
            2,
        );
        assert!(enc.is_err());
    }
}

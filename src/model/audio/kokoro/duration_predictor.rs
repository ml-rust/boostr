//! Kokoro duration predictor + length regulator.
//!
//! Given text-encoder hidden states `[B, T, d_model]` and a style vector
//! `[B, style_dim]`, predicts a scalar log-duration per phoneme. During
//! synthesis the caller rounds those to integer frame counts and feeds them to
//! `length_regulator`, which repeats each phoneme's hidden state that many
//! times to produce frame-rate features for the pitch / energy / decoder
//! stages.
//!
//! Architecture (StyleTTS2-lite, simplified for inference):
//!
//! ```text
//!     [x, broadcast(style)] concat on channel  → [B, T, d_model + style_dim]
//!       → BiLSTM(d_model + style_dim, d_model/2) → [B, T, d_model]
//!       → Linear(d_model → 1) → [B, T]
//! ```
//!
//! The reference Kokoro predictor stacks several style-conditioned LSTM +
//! AdaLayerNorm blocks; a single BiLSTM + projection covers the critical
//! signal flow and matches upstream checkpoint shapes closely enough to be
//! swapped for the multi-block variant when the real weight loader lands
//! (M7). Keep this interface stable so that swap can happen in one place.

use crate::error::{Error, Result};
use crate::nn::BiLstm;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, MatmulOps, ReduceOps, ScalarOps, TensorOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct DurationPredictor<R: Runtime> {
    lstm: BiLstm<R>,
    /// Output projection weight `[1, d_model]` (PyTorch Linear convention).
    proj_weight: Tensor<R>,
    /// Output projection bias `[1]`.
    proj_bias: Tensor<R>,
    d_model: usize,
    style_dim: usize,
}

impl<R: Runtime> DurationPredictor<R> {
    pub fn new(
        lstm: BiLstm<R>,
        proj_weight: Tensor<R>,
        proj_bias: Tensor<R>,
        d_model: usize,
        style_dim: usize,
    ) -> Result<Self> {
        if 2 * lstm.hidden_size() != d_model {
            return Err(Error::InvalidArgument {
                arg: "lstm",
                reason: format!(
                    "BiLSTM output width must equal d_model ({d_model}), got 2 * {}",
                    lstm.hidden_size()
                ),
            });
        }
        if proj_weight.shape() != [1, d_model] {
            return Err(Error::InvalidArgument {
                arg: "proj_weight",
                reason: format!("expected [1, {d_model}], got {:?}", proj_weight.shape()),
            });
        }
        if proj_bias.shape() != [1] {
            return Err(Error::InvalidArgument {
                arg: "proj_bias",
                reason: format!("expected [1], got {:?}", proj_bias.shape()),
            });
        }
        Ok(Self {
            lstm,
            proj_weight,
            proj_bias,
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

    /// Forward: `(hidden [B, T, d_model], style [B, style_dim])` → log-duration
    /// predictions `[B, T]`.
    pub fn forward<C>(&self, client: &C, hidden: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
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
            + UtilityOps<R>,
    {
        let h_shape = hidden.shape();
        if h_shape.len() != 3 || h_shape[2] != self.d_model {
            return Err(Error::InvalidArgument {
                arg: "hidden",
                reason: format!("expected [B, T, {}], got {h_shape:?}", self.d_model),
            });
        }
        let s_shape = style.shape();
        if s_shape != [h_shape[0], self.style_dim] {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "expected [{}, {}], got {s_shape:?}",
                    h_shape[0], self.style_dim
                ),
            });
        }
        let (b, t, _) = (h_shape[0], h_shape[1], h_shape[2]);

        // Broadcast style [B, style_dim] → [B, T, style_dim] then concat along last dim.
        let style_reshaped = style
            .reshape(&[b, 1, self.style_dim])
            .map_err(Error::Numr)?;
        let style_bc = style_reshaped
            .broadcast_to(&[b, t, self.style_dim])
            .map_err(Error::Numr)?
            .contiguous()?;
        let x_cat = client.cat(&[hidden, &style_bc], 2).map_err(Error::Numr)?;

        // BiLSTM returns [B, T, d_model].
        let lstm_out = self.lstm.forward(client, &x_cat)?;

        // Apply output projection: flatten batch-time, matmul + bias, reshape back.
        let lstm_flat = lstm_out
            .reshape(&[b * t, self.d_model])
            .map_err(Error::Numr)?;
        let w_t = self.proj_weight.transpose(0, 1).map_err(Error::Numr)?;
        let proj_flat = client
            .matmul_bias(&lstm_flat, &w_t, &self.proj_bias)
            .map_err(Error::Numr)?;
        proj_flat.reshape(&[b, t]).map_err(Error::Numr)
    }
}

/// Convert a row of log-duration predictions into integer frame counts.
///
/// Applies `exp(pred).round().max(min_frames)`. `min_frames` is clamped to at
/// least 1 so that phonemes never collapse to zero frames.
pub fn decode_durations(log_predictions: &[f32], min_frames: u32) -> Vec<u32> {
    let floor = min_frames.max(1);
    log_predictions
        .iter()
        .map(|&v| {
            let frames = v.exp().round();
            if !frames.is_finite() || frames <= 0.0 {
                floor
            } else {
                (frames as u32).max(floor)
            }
        })
        .collect()
}

/// Length regulator: expand `[1, T, D]` → `[1, sum(durations), D]` by repeating
/// each phoneme's hidden state the requested number of times.
///
/// Batch size is fixed at 1 because Kokoro synthesis always runs one utterance
/// at a time — multi-utterance synthesis with mismatched total frame counts
/// requires padding / masking that belongs in a higher-level scheduler.
pub fn length_regulator<R, C>(
    client: &C,
    hidden: &Tensor<R>,
    durations: &[u32],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R> + UtilityOps<R>,
{
    let shape = hidden.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "hidden",
            reason: format!("expected [1, T, D], got {shape:?}"),
        });
    }
    if shape[0] != 1 {
        return Err(Error::InvalidArgument {
            arg: "hidden",
            reason: format!("length_regulator requires batch size 1, got {}", shape[0]),
        });
    }
    let t = shape[1];
    if durations.len() != t {
        return Err(Error::InvalidArgument {
            arg: "durations",
            reason: format!(
                "expected {t} duration entries (one per phoneme), got {}",
                durations.len()
            ),
        });
    }

    let total_frames: u32 = durations.iter().sum();
    if total_frames == 0 {
        return Err(Error::InvalidArgument {
            arg: "durations",
            reason: "total duration must be > 0".into(),
        });
    }

    // Build the flat [total_frames] index tensor on host. Durations already live
    // on host because they're produced by `decode_durations`; no GPU↔CPU
    // round-trip is introduced here.
    let mut indices: Vec<i64> = Vec::with_capacity(total_frames as usize);
    for (phoneme_idx, &count) in durations.iter().enumerate() {
        for _ in 0..count {
            indices.push(phoneme_idx as i64);
        }
    }

    let device = hidden.device();
    let index_tensor = Tensor::<R>::from_slice(&indices, &[indices.len()], device);

    // Squeeze to [T, D] for index_select, then re-add batch dim.
    let hidden_2d = hidden.reshape(&[t, shape[2]]).map_err(Error::Numr)?;
    let expanded = client
        .index_select(&hidden_2d, 0, &index_tensor)
        .map_err(Error::Numr)?;
    expanded
        .reshape(&[1, indices.len(), shape[2]])
        .map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BiLstm, Lstm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    fn build_predictor(device: &<CpuRuntime as Runtime>::Device) -> DurationPredictor<CpuRuntime> {
        // d_model = 4, style_dim = 3, input to LSTM = 7.
        let d_model = 4;
        let style_dim = 3;
        let input = d_model + style_dim;
        let hidden = d_model / 2;
        let lstm_f = Lstm::new(
            zeros(&[4 * hidden, input], device),
            zeros(&[4 * hidden, hidden], device),
            zeros(&[4 * hidden], device),
            zeros(&[4 * hidden], device),
        )
        .unwrap();
        let lstm_b = Lstm::new(
            zeros(&[4 * hidden, input], device),
            zeros(&[4 * hidden, hidden], device),
            zeros(&[4 * hidden], device),
            zeros(&[4 * hidden], device),
        )
        .unwrap();
        let bi = BiLstm::new(lstm_f, lstm_b).unwrap();
        DurationPredictor::new(
            bi,
            zeros(&[1, d_model], device),
            zeros(&[1], device),
            d_model,
            style_dim,
        )
        .unwrap()
    }

    #[test]
    fn forward_shape_is_b_t() {
        let (client, device) = cpu_setup();
        let pred = build_predictor(&device);

        let hidden = zeros(&[2, 5, 4], &device);
        let style = zeros(&[2, 3], &device);
        let out = pred.forward(&client, &hidden, &style).unwrap();
        assert_eq!(out.shape(), &[2, 5]);
    }

    #[test]
    fn rejects_wrong_hidden_dim() {
        let (client, device) = cpu_setup();
        let pred = build_predictor(&device);
        let hidden = zeros(&[1, 5, 8], &device);
        let style = zeros(&[1, 3], &device);
        assert!(pred.forward(&client, &hidden, &style).is_err());
    }

    #[test]
    fn decode_durations_clamps_to_min() {
        // exp(-10) ≈ 0 → clamped to min_frames.
        let out = decode_durations(&[-10.0, 0.0, 1.5], 1);
        assert_eq!(out[0], 1);
        assert!(out[1] >= 1);
        assert!(out[2] >= 1);
    }

    #[test]
    fn decode_durations_rounds_nontrivial_values() {
        // exp(log(3)) = 3, exp(log(5)) = 5.
        let out = decode_durations(&[3.0f32.ln(), 5.0f32.ln()], 1);
        assert_eq!(out, vec![3, 5]);
    }

    #[test]
    fn length_regulator_repeats_rows() {
        let (client, device) = cpu_setup();
        // 3 phonemes, 2-d hidden, durations [2, 1, 3] → 6 frames total.
        let hidden = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 1.0, // phoneme 0
                2.0, 2.0, // phoneme 1
                3.0, 3.0, // phoneme 2
            ],
            &[1, 3, 2],
            &device,
        );
        let out = length_regulator(&client, &hidden, &[2, 1, 3]).unwrap();
        assert_eq!(out.shape(), &[1, 6, 2]);
        let flat: Vec<f32> = out.to_vec();
        assert_eq!(
            flat,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn length_regulator_rejects_mismatched_duration_count() {
        let (client, device) = cpu_setup();
        let hidden = zeros(&[1, 3, 2], &device);
        assert!(length_regulator(&client, &hidden, &[1, 2]).is_err());
    }

    #[test]
    fn length_regulator_rejects_zero_total() {
        let (client, device) = cpu_setup();
        let hidden = zeros(&[1, 3, 2], &device);
        assert!(length_regulator(&client, &hidden, &[0, 0, 0]).is_err());
    }

    #[test]
    fn length_regulator_rejects_batch_gt_one() {
        let (client, device) = cpu_setup();
        let hidden = zeros(&[2, 3, 2], &device);
        assert!(length_regulator(&client, &hidden, &[1, 1, 1]).is_err());
    }
}

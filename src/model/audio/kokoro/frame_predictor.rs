//! Per-frame scalar predictor — shared scaffold for pitch (F0) and energy.
//!
//! Given frame-rate hidden features `[B, T_frames, d_model]` from the length
//! regulator and a style vector `[B, style_dim]`, predicts one scalar per
//! frame (`[B, T_frames]`). Used twice in the Kokoro pipeline: once for F0 and
//! once for energy, with independent weights.
//!
//! Signal flow mirrors `DurationPredictor`:
//!
//! ```text
//!     [x, broadcast(style)] → concat  → BiLSTM → Linear(d_model → 1) → [B, T]
//! ```
//!
//! The reference Kokoro pitch/energy predictors add style-conditioned Conv1d
//! blocks between the LSTM and projection — same argument as for the duration
//! predictor: same tensor shapes, same constructor signature, swap the block
//! contents when the loader (M7) lands. Holding the interface steady here lets
//! M6 (decoder) be written once against the final I/O shapes.

use crate::error::{Error, Result};
use crate::nn::BiLstm;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, MatmulOps, ReduceOps, ScalarOps, TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct FramePredictor<R: Runtime> {
    lstm: BiLstm<R>,
    /// `[1, d_model]` — PyTorch Linear convention.
    proj_weight: Tensor<R>,
    /// `[1]`.
    proj_bias: Tensor<R>,
    d_model: usize,
    style_dim: usize,
}

impl<R: Runtime> FramePredictor<R> {
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

    /// Forward: `(frames [B, T, d_model], style [B, style_dim])` → `[B, T]`.
    pub fn forward<C>(&self, client: &C, frames: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
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
        let h_shape = frames.shape();
        if h_shape.len() != 3 || h_shape[2] != self.d_model {
            return Err(Error::InvalidArgument {
                arg: "frames",
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

        let style_bc = style
            .reshape(&[b, 1, self.style_dim])
            .map_err(Error::Numr)?
            .broadcast_to(&[b, t, self.style_dim])
            .map_err(Error::Numr)?
            .contiguous();
        let cat = client.cat(&[frames, &style_bc], 2).map_err(Error::Numr)?;

        let lstm_out = self.lstm.forward(client, &cat)?;

        let flat = lstm_out
            .reshape(&[b * t, self.d_model])
            .map_err(Error::Numr)?;
        let w_t = self.proj_weight.transpose(0, 1).map_err(Error::Numr)?;
        let proj = client
            .matmul_bias(&flat, &w_t, &self.proj_bias)
            .map_err(Error::Numr)?;
        proj.reshape(&[b, t]).map_err(Error::Numr)
    }
}

/// Alias: the pitch (F0) predictor shares the scalar-per-frame scaffold.
pub type PitchPredictor<R> = FramePredictor<R>;

/// Alias: the energy predictor shares the scalar-per-frame scaffold.
pub type EnergyPredictor<R> = FramePredictor<R>;

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

    fn build(device: &<CpuRuntime as Runtime>::Device) -> FramePredictor<CpuRuntime> {
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
        FramePredictor::new(
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
        let pred = build(&device);
        let frames = zeros(&[2, 7, 4], &device);
        let style = zeros(&[2, 3], &device);
        let out = pred.forward(&client, &frames, &style).unwrap();
        assert_eq!(out.shape(), &[2, 7]);
    }

    #[test]
    fn pitch_and_energy_aliases_compile() {
        let (_client, device) = cpu_setup();
        let _pitch: PitchPredictor<CpuRuntime> = build(&device);
        let _energy: EnergyPredictor<CpuRuntime> = build(&device);
    }

    #[test]
    fn rejects_wrong_frames_rank() {
        let (client, device) = cpu_setup();
        let pred = build(&device);
        let frames = zeros(&[2, 4], &device);
        let style = zeros(&[2, 3], &device);
        assert!(pred.forward(&client, &frames, &style).is_err());
    }

    #[test]
    fn rejects_wrong_style_shape() {
        let (client, device) = cpu_setup();
        let pred = build(&device);
        let frames = zeros(&[1, 4, 4], &device);
        let style = zeros(&[1, 5], &device);
        assert!(pred.forward(&client, &frames, &style).is_err());
    }

    #[test]
    fn new_rejects_lstm_width_mismatch() {
        let (_client, device) = cpu_setup();
        // d_model=4 but BiLSTM output = 2*3 = 6.
        let hidden = 3;
        let input = 7;
        let lstm_f = Lstm::new(
            zeros(&[4 * hidden, input], &device),
            zeros(&[4 * hidden, hidden], &device),
            zeros(&[4 * hidden], &device),
            zeros(&[4 * hidden], &device),
        )
        .unwrap();
        let lstm_b = Lstm::new(
            zeros(&[4 * hidden, input], &device),
            zeros(&[4 * hidden, hidden], &device),
            zeros(&[4 * hidden], &device),
            zeros(&[4 * hidden], &device),
        )
        .unwrap();
        let bi = BiLstm::new(lstm_f, lstm_b).unwrap();
        assert!(
            FramePredictor::new(bi, zeros(&[1, 4], &device), zeros(&[1], &device), 4, 3,).is_err()
        );
    }
}

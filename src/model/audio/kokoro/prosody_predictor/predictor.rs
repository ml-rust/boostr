use super::duration_encoder::DurationEncoder;
use crate::error::{Error, Result};
use crate::model::audio::kokoro::AdainResBlk1d;
use crate::nn::{BiLstm, Conv1d};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::prosody_predictor::test_support::{
        adaln, bilstm, conv, kadain, zeros,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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
}

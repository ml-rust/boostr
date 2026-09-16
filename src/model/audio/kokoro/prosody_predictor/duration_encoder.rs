use crate::error::{Error, Result};
use crate::model::audio::kokoro::AdaLayerNorm;
use crate::nn::BiLstm;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Alternating `LSTM | AdaLayerNorm` stack (the reference Kokoro
/// implementation's `DurationEncoder`).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::prosody_predictor::test_support::{adaln, bilstm, zeros};
    use crate::test_utils::cpu_setup;

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

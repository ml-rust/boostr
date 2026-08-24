//! `Wav2Vec2BertFeatureProjection` — the semantic branch's entry stage.
//!
//! ```text
//! x [B, T, 160]
//!   -> normed = LayerNorm(eps=1e-5)(x)     # over the INPUT width 160
//!   -> Linear(160 -> 1024)(normed)
//! [B, T, 1024]
//! ```
//!
//! ## The LayerNorm runs BEFORE the projection, over the 160-wide input
//!
//! The plausible-but-wrong alternative is to normalize the projection's
//! *output* over 1024 channels, which is what most "projection + norm" stages
//! in other encoders do. Here it is the other way round, and the checkpoint
//! proves it: `feature_projection.layer_norm.{weight,bias}` are `[160]`, not
//! `[1024]`. Loading them into a 1024-wide norm cannot even be made to
//! typecheck against the real tensors — but hand-rolling the order without
//! looking at the shapes silently produces a differently-normalized encoder.
//!
//! Upstream also returns the intermediate `normed` alongside the projection
//! (it feeds the masked-spec training path). The encoder never consumes it, so
//! this port returns only the projected value.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::{LayerNorm, Linear};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Already-built weights for [`FeatureProjection`].
pub struct FeatureProjectionWeights<R: Runtime> {
    /// `feature_projection.layer_norm.{weight,bias}`, width = input dim (160).
    pub layer_norm: LayerNorm<R>,
    /// `feature_projection.projection.{weight,bias}`, `[1024, 160]` / `[1024]`.
    pub projection: Linear<R>,
}

/// Pre-norm feature projection: `[B, T, in_dim] -> [B, T, hidden]`.
pub struct FeatureProjection<R: Runtime> {
    layer_norm: LayerNorm<R>,
    projection: Linear<R>,
    input_dim: usize,
}

impl<R: Runtime> FeatureProjection<R> {
    /// Assemble from already-loaded weights. `input_dim` is the width the
    /// layer norm and the projection's input axis are checked against.
    pub fn new(weights: FeatureProjectionWeights<R>, input_dim: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input_dim",
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            layer_norm: weights.layer_norm,
            projection: weights.projection,
            input_dim,
        })
    }
}

impl<R: Runtime<DType = DType>> FeatureProjection<R> {
    /// Forward: `x [B, T, in_dim] -> [B, T, hidden]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.input_dim {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected stacked features [B, T, {}], got {shape:?}",
                    self.input_dim
                ),
            });
        }

        // Norm first, over the 160-wide input — see the module doc.
        let normed = self.layer_norm.forward(client, x)?;
        self.projection.forward(client, &normed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn projection(
        in_dim: usize,
        hidden: usize,
        device: &CpuDevice,
    ) -> FeatureProjection<CpuRuntime> {
        let layer_norm = LayerNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; in_dim], &[in_dim], device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; in_dim], &[in_dim], device).unwrap(),
            1e-5,
            false,
        );
        let linear = Linear::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.01f32; hidden * in_dim],
                &[hidden, in_dim],
                device,
            )
            .unwrap(),
            Some(
                Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; hidden], &[hidden], device).unwrap(),
            ),
            false,
        );
        FeatureProjection::new(
            FeatureProjectionWeights {
                layer_norm,
                projection: linear,
            },
            in_dim,
        )
        .expect("build projection")
    }

    #[test]
    fn forward_maps_input_dim_to_hidden() {
        let (client, device) = cpu_setup();
        let (in_dim, hidden, t) = (10, 16, 5);
        let p = projection(in_dim, hidden, &device);

        let data: Vec<f32> = (0..(t * in_dim)).map(|i| (i as f32 * 0.13).sin()).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, t, in_dim], &device).unwrap(),
            false,
        );
        let y = p.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[1, t, hidden]);
    }

    #[test]
    fn rejects_wrong_input_width() {
        let (client, device) = cpu_setup();
        let p = projection(10, 16, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 5 * 7], &[1, 5, 7], &device).unwrap(),
            false,
        );
        assert!(p.forward(&client, &x).is_err());
    }
}

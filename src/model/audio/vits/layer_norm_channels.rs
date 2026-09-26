//! Shared channel-axis `LayerNorm` helper, used by both duration predictors
//! and [`super::dds_conv::VitsDilatedDepthSeparableConv`].

use crate::error::{Error, Result};
use crate::nn::LayerNorm;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Applies `ln` over the channel axis, matching HF's
/// `norm(x.transpose(1, -1)).transpose(1, -1)`: `[B, C, T] -> transpose(1,2)
/// -> LayerNorm over C -> transpose back -> [B, C, T]`.
pub(in crate::model::audio::vits) fn layer_norm_channels<R, C>(
    client: &C,
    ln: &LayerNorm<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let t = x
        .transpose(1, 2)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?;
    let out = ln.forward(client, &Var::new(t, false))?;
    out.tensor()
        .transpose(1, 2)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn layer_norm_channels_normalizes_over_channel_axis_not_time() {
        let (client, device) = cpu_setup();
        // [B=1, C=4, T=2]: two time steps, each with a distinct channel
        // profile. If the transpose direction were wrong, this would
        // normalize over T=2 instead of C=4 and the shapes would still
        // match but the values would differ from a manual per-time-step
        // channel normalization.
        // Row-major [1, C=4, T=2]: element (c, t) sits at flat index c*2+t.
        // t=0 profile is [1,2,3,4] across channels, t=1 profile is
        // [10,20,30,40] — a positive rescale of the same shape.
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            &[1, 4, 2],
            &device,
        )
        .unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        let ln = LayerNorm::new(weight, bias, 1e-5, false);

        let out = layer_norm_channels(&client, &ln, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        let data: Vec<f32> = out.to_vec();
        // Each time step's 4 channel values are independently normalized to
        // zero mean; column 0 (t=0) and column 1 (t=1) should therefore be
        // numerically identical since [1,2,3,4] and [10,20,30,40] are the
        // same shape up to an affine rescale.
        for c in 0..4 {
            let a = data[c * 2];
            let b = data[c * 2 + 1];
            assert!((a - b).abs() < 1e-3, "channel {c}: {a} vs {b}");
        }
    }
}

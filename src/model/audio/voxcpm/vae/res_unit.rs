//! `ResUnit`: the dilated residual block repeated 3x (dilations 1, 3, 9)
//! inside every VoxCPM2 `DecoderBlock`.
//!
//! `y = Snake(dim) -> CausalConv1d(dim->dim, k=7, dilation, groups=dim) ->
//! Snake(dim) -> CausalConv1d(dim->dim, k=1, groups=1)`, then `x + y`.
//!
//! Both convs are causal and length-preserving, so the residual add never
//! needs cropping; a length mismatch here means the causal padding math is
//! wrong upstream, so it is treated as a hard error rather than silently
//! cropped.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Weights for one `ResUnit`.
pub struct ResUnit<R: Runtime> {
    snake1: Snake<R>,
    /// Depthwise, `k=7`, `groups=dim`, dilated.
    dilated_conv: CausalConv1d<R>,
    snake2: Snake<R>,
    /// Pointwise, `k=1`, `groups=1`.
    pointwise_conv: CausalConv1d<R>,
}

impl<R: Runtime<DType = DType>> ResUnit<R> {
    pub fn new(
        snake1: Snake<R>,
        dilated_conv: CausalConv1d<R>,
        snake2: Snake<R>,
        pointwise_conv: CausalConv1d<R>,
    ) -> Self {
        Self {
            snake1,
            dilated_conv,
            snake2,
            pointwise_conv,
        }
    }

    /// `x [B, C, T] -> [B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let h = self.snake1.forward(client, x)?;
        let h = self.dilated_conv.forward(client, &h)?;
        let h = self.snake2.forward(client, &h)?;
        let h = self.pointwise_conv.forward(client, &h)?;

        if h.shape() != x.shape() {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "ResUnit output shape {:?} does not match input shape {:?}; causal padding \
                     must keep length exact, cropping is not performed",
                    h.shape(),
                    x.shape()
                ),
            });
        }
        client.add(x, &h).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn snake(c: usize, device: &<CpuRuntime as Runtime>::Device) -> Snake<CpuRuntime> {
        let alpha = Tensor::<CpuRuntime>::from_slice(&vec![0.5f32; c], &[1, c, 1], device).unwrap();
        Snake::new(alpha).unwrap()
    }

    fn dilated_conv(
        c: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let k = 7;
        let weight =
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * k], &[c, 1, k], device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap();
        CausalConv1d::new(weight, Some(bias), k, dilation, c).unwrap()
    }

    fn pointwise_conv(
        c: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let weight =
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * c], &[c, c, 1], device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap();
        CausalConv1d::new(weight, Some(bias), 1, 1, 1).unwrap()
    }

    #[test]
    fn preserves_shape_and_is_finite() {
        let (client, device) = cpu_setup();
        let c = 4;
        let t = 12;
        let unit = ResUnit::new(
            snake(c, &device),
            dilated_conv(c, 3, &device),
            snake(c, &device),
            pointwise_conv(c, &device),
        );
        let x_data: Vec<f32> = (0..(c * t)).map(|i| (i as f32 * 0.1).sin()).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, c, t], &device).unwrap();
        let out = unit.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), x.shape());
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }
}

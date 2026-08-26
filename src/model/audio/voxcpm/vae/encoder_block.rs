//! One VoxCPM2 `AudioVAE` encoder stage: three dilated `ResUnit`s, then
//! `Snake`, then a strided downsampling `CausalConv1d`.
//!
//! This is the REVERSE order of the decoder's `DecoderBlock` (which upsamples
//! first, then runs the `ResUnit`s) and carries no sample-rate conditioning —
//! that machinery is decoder-only.
//!
//! `x = ResUnit(d=1) -> ResUnit(d=3) -> ResUnit(d=9)`
//! `x = Snake(input_dim) -> CausalConv1d::new_strided(input_dim->output_dim, stride)`

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Weights for one `EncoderBlock`.
pub struct EncoderBlockWeights<R: Runtime> {
    pub res1: ResUnit<R>,
    pub res3: ResUnit<R>,
    pub res9: ResUnit<R>,
    pub snake: Snake<R>,
    /// Strided, `input_dim -> output_dim`, `kernel = 2 * stride`.
    pub downsample: CausalConv1d<R>,
}

/// One `EncoderBlock` stage of the `AudioVaeEncoder`.
pub struct EncoderBlock<R: Runtime> {
    res1: ResUnit<R>,
    res3: ResUnit<R>,
    res9: ResUnit<R>,
    snake: Snake<R>,
    downsample: CausalConv1d<R>,
    input_dim: usize,
}

impl<R: Runtime<DType = DType>> EncoderBlock<R> {
    pub fn new(weights: EncoderBlockWeights<R>) -> Self {
        let input_dim = weights.snake.channels();
        Self {
            res1: weights.res1,
            res3: weights.res3,
            res9: weights.res9,
            snake: weights.snake,
            downsample: weights.downsample,
            input_dim,
        }
    }

    /// `x [B, input_dim, T] -> [B, output_dim, T / stride]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != self.input_dim {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {shape:?}", self.input_dim),
            });
        }

        let x = self.res1.forward(client, x)?;
        let x = self.res3.forward(client, &x)?;
        let x = self.res9.forward(client, &x)?;
        let x = self.snake.forward(client, &x)?;
        self.downsample.forward(client, &x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn snake(c: usize, device: &<CpuRuntime as Runtime>::Device) -> Snake<CpuRuntime> {
        let alpha = Tensor::<CpuRuntime>::from_slice(&vec![0.3f32; c], &[1, c, 1], device).unwrap();
        Snake::new(alpha).unwrap()
    }

    fn res_unit(
        c: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> ResUnit<CpuRuntime> {
        let k = 7;
        let dw_weight =
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * k], &[c, 1, k], device).unwrap();
        let dw_bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap();
        let dilated = CausalConv1d::new(dw_weight, Some(dw_bias), k, dilation, c).unwrap();
        let pw_weight =
            Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * c], &[c, c, 1], device).unwrap();
        let pw_bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap();
        let pointwise = CausalConv1d::new(pw_weight, Some(pw_bias), 1, 1, 1).unwrap();
        ResUnit::new(snake(c, device), dilated, snake(c, device), pointwise)
    }

    fn build_block(
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> EncoderBlock<CpuRuntime> {
        let k = 2 * stride;
        let down_weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; output_dim * input_dim * k],
            &[output_dim, input_dim, k],
            device,
        )
        .unwrap();
        let down_bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
                .unwrap();
        EncoderBlock::new(EncoderBlockWeights {
            res1: res_unit(input_dim, 1, device),
            res3: res_unit(input_dim, 3, device),
            res9: res_unit(input_dim, 9, device),
            snake: snake(input_dim, device),
            downsample: CausalConv1d::new_strided(down_weight, Some(down_bias), stride, 1).unwrap(),
        })
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let (client, device) = cpu_setup();
        let (input_dim, output_dim, stride) = (4, 8, 2);
        let block = build_block(input_dim, output_dim, stride, &device);
        let t = stride * 6;
        let x_data: Vec<f32> = (0..(input_dim * t))
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, input_dim, t], &device).unwrap();
        let out = block.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, output_dim, t / stride]);
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_wrong_input_dim() {
        let (client, device) = cpu_setup();
        let block = build_block(4, 8, 2, &device);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8 * 4], &[1, 8, 4], &device).unwrap();
        assert!(block.forward(&client, &x).is_err());
    }
}

//! One VoxCPM2 decoder stage: sample-rate conditioning, upsample, then three
//! dilated `ResUnit`s.
//!
//! `x = x * scale_embed[bucket] + bias_embed[bucket]`
//! `x = Snake(input_dim) -> CausalTransposeConv1d(input_dim->output_dim, stride)`
//! `x = ResUnit(d=1) -> ResUnit(d=3) -> ResUnit(d=9)`

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::causal_transpose_conv1d::CausalTransposeConv1d;
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Weights for one `DecoderBlock`.
pub struct DecoderBlockWeights<R: Runtime> {
    pub snake: Snake<R>,
    pub upsample: CausalTransposeConv1d<R>,
    pub res1: ResUnit<R>,
    pub res3: ResUnit<R>,
    pub res9: ResUnit<R>,
    /// `[num_sr_buckets, input_dim]`.
    pub scale_embed: Tensor<R>,
    /// `[num_sr_buckets, input_dim]`.
    pub bias_embed: Tensor<R>,
}

/// One `DecoderBlock` stage of the `AudioVaeDecoder`.
pub struct DecoderBlock<R: Runtime> {
    snake: Snake<R>,
    upsample: CausalTransposeConv1d<R>,
    res1: ResUnit<R>,
    res3: ResUnit<R>,
    res9: ResUnit<R>,
    scale_embed: Tensor<R>,
    bias_embed: Tensor<R>,
    input_dim: usize,
}

impl<R: Runtime<DType = DType>> DecoderBlock<R> {
    pub fn new(weights: DecoderBlockWeights<R>) -> Result<Self> {
        let input_dim = weights.snake.channels();
        for (name, t) in [
            ("scale_embed", &weights.scale_embed),
            ("bias_embed", &weights.bias_embed),
        ] {
            if t.shape().len() != 2 || t.shape()[1] != input_dim {
                return Err(Error::InvalidArgument {
                    arg: name,
                    reason: format!(
                        "expected [num_sr_buckets, {input_dim}], got {:?}",
                        t.shape()
                    ),
                });
            }
        }
        Ok(Self {
            snake: weights.snake,
            upsample: weights.upsample,
            res1: weights.res1,
            res3: weights.res3,
            res9: weights.res9,
            scale_embed: weights.scale_embed,
            bias_embed: weights.bias_embed,
            input_dim,
        })
    }

    /// `x [B, input_dim, T] -> [B, output_dim, T * stride]`, conditioned on
    /// `sr_bucket` (an index into `scale_embed`/`bias_embed`).
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, sr_bucket: usize) -> Result<Tensor<R>>
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
        let num_buckets = self.scale_embed.shape()[0];
        if sr_bucket >= num_buckets {
            return Err(Error::InvalidArgument {
                arg: "sr_bucket",
                reason: format!("must be < {num_buckets}, got {sr_bucket}"),
            });
        }

        let scale = self
            .scale_embed
            .narrow(0, sr_bucket, 1)
            .map_err(Error::Numr)?
            .reshape(&[1, self.input_dim, 1])
            .map_err(Error::Numr)?;
        let bias = self
            .bias_embed
            .narrow(0, sr_bucket, 1)
            .map_err(Error::Numr)?
            .reshape(&[1, self.input_dim, 1])
            .map_err(Error::Numr)?;

        let x = client.mul(x, &scale).map_err(Error::Numr)?;
        let x = client.add(&x, &bias).map_err(Error::Numr)?;

        let x = self.snake.forward(client, &x)?;
        let x = self.upsample.forward(client, &x)?;
        let x = self.res1.forward(client, &x)?;
        let x = self.res3.forward(client, &x)?;
        self.res9.forward(client, &x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
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
    ) -> DecoderBlock<CpuRuntime> {
        let k = 2 * stride;
        let up_weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; input_dim * output_dim * k],
            &[input_dim, output_dim, k],
            device,
        )
        .unwrap();
        let up_bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
                .unwrap();
        let num_buckets = 4;
        let scale_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![1.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .unwrap();
        let bias_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .unwrap();
        DecoderBlock::new(DecoderBlockWeights {
            snake: snake(input_dim, device),
            upsample: CausalTransposeConv1d::new(up_weight, Some(up_bias), stride).unwrap(),
            res1: res_unit(output_dim, 1, device),
            res3: res_unit(output_dim, 3, device),
            res9: res_unit(output_dim, 9, device),
            scale_embed,
            bias_embed,
        })
        .unwrap()
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let (client, device) = cpu_setup();
        let (input_dim, output_dim, stride) = (8, 4, 2);
        let block = build_block(input_dim, output_dim, stride, &device);
        let t = 6;
        let x_data: Vec<f32> = (0..(input_dim * t))
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, input_dim, t], &device).unwrap();
        let out = block.forward(&client, &x, 3).unwrap();
        assert_eq!(out.shape(), &[1, output_dim, t * stride]);
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_out_of_range_bucket() {
        let (client, device) = cpu_setup();
        let block = build_block(8, 4, 2, &device);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8 * 3], &[1, 8, 3], &device).unwrap();
        assert!(block.forward(&client, &x, 99).is_err());
    }
}

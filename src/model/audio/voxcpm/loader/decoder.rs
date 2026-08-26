//! Loads `decoder.*` tensors and assembles an [`AudioVaeDecoder`].
//!
//! Verified key layout (already weight-norm folded — `.weight`/`.bias` only,
//! no `weight_g`/`weight_v`):
//!
//! ```text
//! decoder.model.0.{weight[64,1,7],bias[64]}                  front depthwise
//! decoder.model.1.{weight[2048,64,1],bias[2048]}              front pointwise
//! decoder.model.{2..7}.block.0.alpha            [1,in,1]      DecoderBlock Snake
//! decoder.model.{2..7}.block.1.{weight[in,out,2*stride],bias[out]}   upsample
//! decoder.model.{2..7}.block.{2,3,4}.block.0.alpha   [1,out,1]       ResUnit Snake 1
//! decoder.model.{2..7}.block.{2,3,4}.block.1.{weight[out,1,7],bias[out]}   dilated dw
//! decoder.model.{2..7}.block.{2,3,4}.block.2.alpha   [1,out,1]       ResUnit Snake 2
//! decoder.model.{2..7}.block.{2,3,4}.block.3.{weight[out,out,1],bias[out]} pointwise
//! decoder.model.8.alpha [1,32,1]                              final Snake
//! decoder.model.9.{weight[1,32,7],bias[1]}                    final conv
//! decoder.sr_cond_model.{2..7}.scale_embed.weight [4,in]
//! decoder.sr_cond_model.{2..7}.bias_embed.weight  [4,in]
//! ```
//!
//! `decoder.sr_bin_boundaries` (`[3]`, I32) is NOT loaded: this port always
//! decodes at the fixed [`crate::model::audio::voxcpm::decoder::DEFAULT_SR_BUCKET`]
//! (see that constant's doc comment), so the boundaries used to derive a
//! bucket from an arbitrary target rate are never consulted.

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::causal_transpose_conv1d::CausalTransposeConv1d;
use crate::model::audio::voxcpm::decoder::{
    AudioVaeDecoder, AudioVaeDecoderWeights, CAUSAL_KERNEL, FINAL_CHANNELS, FRONT_HIDDEN,
    INPUT_CHANNELS, NUM_SR_BUCKETS, OUTPUT_CHANNELS, RES_UNIT_DILATIONS, STRIDES,
};
use crate::model::audio::voxcpm::decoder_block::{DecoderBlock, DecoderBlockWeights};
use crate::model::audio::voxcpm::res_unit::ResUnit;
use crate::model::audio::voxcpm::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Default top-level prefix for the `AudioVAE` decoder's tensors in the
/// checkpoint.
pub const DEFAULT_DECODER_PREFIX: &str = "decoder";

/// Per-stage channel widths, `(input_dim, output_dim)`, outermost first —
/// `2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32`.
fn block_dims() -> [(usize, usize); 6] {
    std::array::from_fn(|i| (FRONT_HIDDEN >> i, FRONT_HIDDEN >> (i + 1)))
}

/// Reads `decoder.*` tensors and assembles an [`AudioVaeDecoder`].
struct DecoderLoader<'a, R: Runtime<DType = DType>> {
    loader: &'a mut SafeTensorsLoader,
    device: &'a R::Device,
    prefix: String,
}

impl<R: Runtime<DType = DType>> DecoderLoader<'_, R> {
    fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        checked_tensor::<R>(self.loader, self.device, &self.prefix, name, expected)
    }

    fn snake(&mut self, name: &str, channels: usize) -> Result<Snake<R>> {
        let alpha = self.tensor(&format!("{name}.alpha"), &[1, channels, 1])?;
        Snake::new(alpha)
    }

    /// Depthwise causal conv: `[channels, 1, kernel]`.
    fn depthwise_conv(
        &mut self,
        name: &str,
        channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<CausalConv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[channels, 1, kernel])?;
        let bias = self.tensor(&format!("{name}.bias"), &[channels])?;
        CausalConv1d::new(weight, Some(bias), kernel, dilation, channels)
    }

    /// Pointwise (`k=1`, `groups=1`) causal conv: `[out, in, 1]`.
    fn pointwise_conv(&mut self, name: &str, in_c: usize, out_c: usize) -> Result<CausalConv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_c, in_c, 1])?;
        let bias = self.tensor(&format!("{name}.bias"), &[out_c])?;
        CausalConv1d::new(weight, Some(bias), 1, 1, 1)
    }

    fn res_unit(&mut self, name: &str, dim: usize, dilation: usize) -> Result<ResUnit<R>> {
        let snake1 = self.snake(&format!("{name}.block.0"), dim)?;
        let dilated_conv =
            self.depthwise_conv(&format!("{name}.block.1"), dim, CAUSAL_KERNEL, dilation)?;
        let snake2 = self.snake(&format!("{name}.block.2"), dim)?;
        let pointwise_conv = self.pointwise_conv(&format!("{name}.block.3"), dim, dim)?;
        Ok(ResUnit::new(snake1, dilated_conv, snake2, pointwise_conv))
    }

    fn decoder_block(
        &mut self,
        model_idx: usize,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
    ) -> Result<DecoderBlock<R>> {
        let block_name = format!("model.{model_idx}.block");
        let snake = self.snake(&format!("{block_name}.0"), input_dim)?;

        let up_weight = self.tensor(
            &format!("{block_name}.1.weight"),
            &[input_dim, output_dim, 2 * stride],
        )?;
        let up_bias = self.tensor(&format!("{block_name}.1.bias"), &[output_dim])?;
        let upsample = CausalTransposeConv1d::new(up_weight, Some(up_bias), stride)?;

        let res1 = self.res_unit(
            &format!("{block_name}.2"),
            output_dim,
            RES_UNIT_DILATIONS[0],
        )?;
        let res3 = self.res_unit(
            &format!("{block_name}.3"),
            output_dim,
            RES_UNIT_DILATIONS[1],
        )?;
        let res9 = self.res_unit(
            &format!("{block_name}.4"),
            output_dim,
            RES_UNIT_DILATIONS[2],
        )?;

        let sr_prefix = format!("sr_cond_model.{model_idx}");
        let scale_embed = self.tensor(
            &format!("{sr_prefix}.scale_embed.weight"),
            &[NUM_SR_BUCKETS, input_dim],
        )?;
        let bias_embed = self.tensor(
            &format!("{sr_prefix}.bias_embed.weight"),
            &[NUM_SR_BUCKETS, input_dim],
        )?;

        DecoderBlock::new(DecoderBlockWeights {
            snake,
            upsample,
            res1,
            res3,
            res9,
            scale_embed,
            bias_embed,
        })
    }

    fn build(&mut self) -> Result<AudioVaeDecoderWeights<R>> {
        let front_dw = self.depthwise_conv("model.0", INPUT_CHANNELS, CAUSAL_KERNEL, 1)?;
        let front_pw = self.pointwise_conv("model.1", INPUT_CHANNELS, FRONT_HIDDEN)?;

        let dims = block_dims();
        let mut blocks_vec = Vec::with_capacity(6);
        for (i, (input_dim, output_dim)) in dims.into_iter().enumerate() {
            blocks_vec.push(self.decoder_block(i + 2, input_dim, output_dim, STRIDES[i])?);
        }
        let blocks: [DecoderBlock<R>; 6] = blocks_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("exactly 6 DecoderBlocks were pushed"));

        let final_snake = self.snake("model.8", FINAL_CHANNELS)?;
        let final_conv = {
            let weight = self.tensor(
                "model.9.weight",
                &[OUTPUT_CHANNELS, FINAL_CHANNELS, CAUSAL_KERNEL],
            )?;
            let bias = self.tensor("model.9.bias", &[OUTPUT_CHANNELS])?;
            CausalConv1d::new(weight, Some(bias), CAUSAL_KERNEL, 1, 1)?
        };

        Ok(AudioVaeDecoderWeights {
            front_dw,
            front_pw,
            blocks,
            final_snake,
            final_conv,
        })
    }
}

impl<R: Runtime<DType = DType>> AudioVaeDecoder<R> {
    /// Load the `AudioVAE` decoder from a VoxCPM2 checkpoint.
    ///
    /// `path` may be either the `model.safetensors` file or the directory
    /// containing it.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with(path, DEFAULT_DECODER_PREFIX, device)
    }

    /// Load with an explicit checkpoint prefix.
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        prefix: &str,
        device: &R::Device,
    ) -> Result<Self> {
        let mut loader = SafeTensorsLoader::open(path)?;
        let weights = DecoderLoader::<R> {
            loader: &mut loader,
            device,
            prefix: prefix.to_string(),
        }
        .build()?;
        Ok(Self::new(weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            AudioVaeDecoder::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                &device
            )
            .is_err()
        );
    }
}

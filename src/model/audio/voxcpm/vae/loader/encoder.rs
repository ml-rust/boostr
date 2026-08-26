//! Loads `encoder.*` tensors and assembles an [`AudioVaeEncoder`].
//!
//! Verified key layout (already weight-norm folded — `.weight`/`.bias` only,
//! no `weight_g`/`weight_v`):
//!
//! ```text
//! encoder.block.0.{weight[128,1,7],bias[128]}                        front conv
//! encoder.block.{1..4}.block.{0,1,2}.block.0.alpha   [1,in,1]         ResUnit Snake 1
//! encoder.block.{1..4}.block.{0,1,2}.block.1.{weight[in,1,7],bias[in]}   dilated dw
//! encoder.block.{1..4}.block.{0,1,2}.block.2.alpha   [1,in,1]         ResUnit Snake 2
//! encoder.block.{1..4}.block.{0,1,2}.block.3.{weight[in,in,1],bias[in]}  pointwise
//! encoder.block.{1..4}.block.3.alpha                 [1,in,1]         EncoderBlock Snake
//! encoder.block.{1..4}.block.4.{weight[out,in,2*stride],bias[out]}    downsample
//! encoder.fc_mu.{weight[64,2048,3],bias[64]}
//! ```
//!
//! `encoder.fc_logvar` (same shape as `fc_mu`) is NOT loaded: `AudioVAE.encode()`
//! returns `mu` only, deterministically, with no reparameterisation sampling
//! (see [`crate::model::audio::voxcpm::vae::encoder`] module docs).

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::loader::support::TensorLoader;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::encoder::{
    AudioVaeEncoder, AudioVaeEncoderWeights, FINAL_HIDDEN, FRONT_HIDDEN, HEAD_KERNEL,
    INPUT_CHANNELS, OUTPUT_CHANNELS, RES_KERNEL, RES_UNIT_DILATIONS, STRIDES,
};
use crate::model::audio::voxcpm::vae::encoder_block::{EncoderBlock, EncoderBlockWeights};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Default top-level prefix for the `AudioVAE` encoder's tensors in the
/// checkpoint.
pub const DEFAULT_ENCODER_PREFIX: &str = "encoder";

/// Per-stage channel widths, `(input_dim, output_dim)`, in forward order —
/// `128 -> 256 -> 512 -> 1024 -> 2048`.
fn block_dims() -> [(usize, usize); 4] {
    std::array::from_fn(|i| (FRONT_HIDDEN << i, FRONT_HIDDEN << (i + 1)))
}

/// Reads `encoder.*` tensors and assembles an [`AudioVaeEncoder`]. The
/// tensor/snake/conv/`ResUnit` reads themselves are shared with the decoder
/// loader via [`TensorLoader`]; only the block/front/head assembly below is
/// encoder-specific.
type EncoderLoader<'a, R> = TensorLoader<'a, R>;

impl<R: Runtime<DType = DType>> EncoderLoader<'_, R>
where
    R::Client: TypeConversionOps<R>,
{
    fn encoder_block(
        &mut self,
        model_idx: usize,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
    ) -> Result<EncoderBlock<R>> {
        let block_name = format!("block.{model_idx}.block");

        let res1 = self.res_unit(
            &format!("{block_name}.0"),
            input_dim,
            RES_KERNEL,
            RES_UNIT_DILATIONS[0],
        )?;
        let res3 = self.res_unit(
            &format!("{block_name}.1"),
            input_dim,
            RES_KERNEL,
            RES_UNIT_DILATIONS[1],
        )?;
        let res9 = self.res_unit(
            &format!("{block_name}.2"),
            input_dim,
            RES_KERNEL,
            RES_UNIT_DILATIONS[2],
        )?;

        let snake = self.snake(&format!("{block_name}.3"), input_dim)?;

        let down_weight = self.tensor(
            &format!("{block_name}.4.weight"),
            &[output_dim, input_dim, 2 * stride],
        )?;
        let down_bias = self.tensor(&format!("{block_name}.4.bias"), &[output_dim])?;
        let downsample = CausalConv1d::new_strided(down_weight, Some(down_bias), stride, 1)?;

        Ok(EncoderBlock::new(EncoderBlockWeights {
            res1,
            res3,
            res9,
            snake,
            downsample,
        }))
    }

    fn build_encoder(&mut self) -> Result<AudioVaeEncoderWeights<R>> {
        let front = {
            let weight = self.tensor(
                "block.0.weight",
                &[FRONT_HIDDEN, INPUT_CHANNELS, RES_KERNEL],
            )?;
            let bias = self.tensor("block.0.bias", &[FRONT_HIDDEN])?;
            CausalConv1d::new(weight, Some(bias), RES_KERNEL, 1, 1)?
        };

        let dims = block_dims();
        let mut blocks_vec = Vec::with_capacity(4);
        for (i, (input_dim, output_dim)) in dims.into_iter().enumerate() {
            blocks_vec.push(self.encoder_block(i + 1, input_dim, output_dim, STRIDES[i])?);
        }
        let got = blocks_vec.len();
        let blocks: [EncoderBlock<R>; 4] =
            blocks_vec.try_into().map_err(|_| Error::ModelError {
                reason: format!("expected 4 EncoderBlocks, assembled {got}"),
            })?;

        let fc_mu = {
            let weight = self.tensor(
                "fc_mu.weight",
                &[OUTPUT_CHANNELS, FINAL_HIDDEN, HEAD_KERNEL],
            )?;
            let bias = self.tensor("fc_mu.bias", &[OUTPUT_CHANNELS])?;
            CausalConv1d::new(weight, Some(bias), HEAD_KERNEL, 1, 1)?
        };

        Ok(AudioVaeEncoderWeights {
            front,
            blocks,
            fc_mu,
        })
    }
}

impl<R: Runtime<DType = DType>> AudioVaeEncoder<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load the `AudioVAE` encoder from a VoxCPM2 checkpoint.
    ///
    /// `path` may be either the `model.safetensors` file or the directory
    /// containing it.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with(path, DEFAULT_ENCODER_PREFIX, device)
    }

    /// Load with an explicit checkpoint prefix.
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        prefix: &str,
        device: &R::Device,
    ) -> Result<Self> {
        let mut loader = SafeTensorsLoader::open(path)?;
        let weights = EncoderLoader::<R> {
            loader: &mut loader,
            device,
            prefix: prefix.to_string(),
            dtype: None,
        }
        .build_encoder()?;
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
            AudioVaeEncoder::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                &device
            )
            .is_err()
        );
    }
}

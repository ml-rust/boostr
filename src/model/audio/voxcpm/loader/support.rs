//! Shared tensor-fetch and sub-module-assembly helpers for the VoxCPM2
//! encoder/decoder loaders.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::res_unit::ResUnit;
use crate::model::audio::voxcpm::snake::Snake;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Load `{prefix}.{name}` and verify its shape matches `expected`.
pub(super) fn checked_tensor<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    device: &R::Device,
    prefix: &str,
    name: &str,
    expected: &[usize],
) -> Result<Tensor<R>> {
    let full = format!("{prefix}.{name}");
    let t = loader.load_tensor::<R>(&full, device)?;
    if t.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{full}: expected shape {expected:?}, checkpoint has {:?}",
                t.shape()
            ),
        });
    }
    Ok(t)
}

/// Checkpoint-tensor reader shared by the encoder and decoder loaders: both
/// walk the same `Snake -> depthwise CausalConv1d -> Snake -> pointwise
/// CausalConv1d` `ResUnit` layout, just under different key prefixes and
/// kernel-size constants, so that walk lives here once. `encoder.rs` and
/// `decoder.rs` each add their own inherent `impl` (block/front/head
/// assembly) on this same type for their block-specific layout.
pub(super) struct TensorLoader<'a, R: Runtime<DType = DType>> {
    pub(super) loader: &'a mut SafeTensorsLoader,
    pub(super) device: &'a R::Device,
    pub(super) prefix: String,
}

impl<R: Runtime<DType = DType>> TensorLoader<'_, R> {
    pub(super) fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        checked_tensor::<R>(self.loader, self.device, &self.prefix, name, expected)
    }

    pub(super) fn snake(&mut self, name: &str, channels: usize) -> Result<Snake<R>> {
        let alpha = self.tensor(&format!("{name}.alpha"), &[1, channels, 1])?;
        Snake::new(alpha)
    }

    /// Depthwise causal conv: `[channels, 1, kernel]`.
    pub(super) fn depthwise_conv(
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
    pub(super) fn pointwise_conv(
        &mut self,
        name: &str,
        in_c: usize,
        out_c: usize,
    ) -> Result<CausalConv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_c, in_c, 1])?;
        let bias = self.tensor(&format!("{name}.bias"), &[out_c])?;
        CausalConv1d::new(weight, Some(bias), 1, 1, 1)
    }

    pub(super) fn res_unit(
        &mut self,
        name: &str,
        dim: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<ResUnit<R>> {
        let snake1 = self.snake(&format!("{name}.block.0"), dim)?;
        let dilated_conv =
            self.depthwise_conv(&format!("{name}.block.1"), dim, kernel, dilation)?;
        let snake2 = self.snake(&format!("{name}.block.2"), dim)?;
        let pointwise_conv = self.pointwise_conv(&format!("{name}.block.3"), dim, dim)?;
        Ok(ResUnit::new(snake1, dilated_conv, snake2, pointwise_conv))
    }
}

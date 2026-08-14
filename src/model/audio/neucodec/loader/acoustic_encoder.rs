//! Loads `acoustic_encoder.*` and assembles an [`AcousticEncoder`].

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::neucodec::acoustic_encoder::{
    AcousticEncoder, AcousticEncoderWeights, ENCODER_BASE_CHANNELS, ENCODER_STRIDES, EncoderBlock,
    EncoderBlockWeights, RESIDUAL_DILATIONS, RESIDUAL_KERNEL_SIZE, ResidualUnit,
    ResidualUnitWeights, downsample_padding, same_padding,
};
use crate::model::audio::neucodec::alias_free::{Activation1d, SnakeBeta};
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Top-level prefix for the acoustic (BigCodec) encoder.
pub const DEFAULT_ACOUSTIC_ENCODER_PREFIX: &str = "acoustic_encoder";

/// Output width of the acoustic encoder's final conv (checkpoint: 1024).
const ACOUSTIC_ENCODER_OUT_DIM: usize = 1024;

/// Reads `acoustic_encoder.*` and assembles an [`AcousticEncoder`].
struct AcousticEncoderLoader<'a, R: Runtime<DType = DType>> {
    loader: &'a mut SafeTensorsLoader,
    device: &'a R::Device,
    prefix: String,
}

impl<R: Runtime<DType = DType>> AcousticEncoderLoader<'_, R> {
    fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        checked_tensor::<R>(self.loader, self.device, &self.prefix, name, expected)
    }

    /// `Activation1d` wrapping a `SnakeBeta`; alpha/beta are log-scale `[C]`.
    fn activation(&mut self, name: &str, channels: usize) -> Result<Activation1d<R>> {
        let alpha = self.tensor(&format!("{name}.act.alpha"), &[channels])?;
        let beta = self.tensor(&format!("{name}.act.beta"), &[channels])?;
        let snake = SnakeBeta::new(alpha, beta, false)?;
        Activation1d::new(snake, self.device)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        name: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Result<Conv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_ch, in_ch, kernel])?;
        let bias = self.tensor(&format!("{name}.bias"), &[out_ch])?;
        Ok(Conv1d::new(
            weight,
            Some(bias),
            stride,
            PaddingMode::Custom(pad, pad, 0, 0),
            dilation,
            1,
            false,
        ))
    }

    fn residual_unit(
        &mut self,
        name: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<ResidualUnit<R>> {
        let pad = same_padding(RESIDUAL_KERNEL_SIZE, dilation);
        Ok(ResidualUnit::new(ResidualUnitWeights {
            snake1: self.activation(&format!("{name}.snake1"), channels)?,
            conv1: self.conv(
                &format!("{name}.conv1"),
                channels,
                channels,
                RESIDUAL_KERNEL_SIZE,
                1,
                pad,
                dilation,
            )?,
            snake2: self.activation(&format!("{name}.snake2"), channels)?,
            conv2: self.conv(&format!("{name}.conv2"), channels, channels, 1, 1, 0, 1)?,
        }))
    }

    fn block(&mut self, idx: usize, in_ch: usize, stride: usize) -> Result<EncoderBlock<R>> {
        let name = format!("block.{idx}");
        let out_ch = in_ch * 2;
        let mut res_units = Vec::with_capacity(RESIDUAL_DILATIONS.len());
        for (i, dilation) in RESIDUAL_DILATIONS.iter().enumerate() {
            // Upstream names them res_unit1..3 (1-based).
            res_units.push(self.residual_unit(
                &format!("{name}.res_unit{}", i + 1),
                in_ch,
                *dilation,
            )?);
        }
        EncoderBlock::new(EncoderBlockWeights {
            res_units,
            snake1: self.activation(&format!("{name}.snake1"), in_ch)?,
            conv1: self.conv(
                &format!("{name}.conv1"),
                out_ch,
                in_ch,
                2 * stride,
                stride,
                downsample_padding(stride),
                1,
            )?,
        })
    }

    fn build(&mut self) -> Result<AcousticEncoderWeights<R>> {
        let mut channels = ENCODER_BASE_CHANNELS;
        let conv1 = self.conv("conv1", channels, 1, 7, 1, 3, 1)?;

        let mut blocks = Vec::with_capacity(ENCODER_STRIDES.len());
        for (idx, stride) in ENCODER_STRIDES.iter().enumerate() {
            blocks.push(self.block(idx, channels, *stride)?);
            channels *= 2;
        }

        let snake1 = self.activation("snake1", channels)?;
        let conv2 = self.conv("conv2", ACOUSTIC_ENCODER_OUT_DIM, channels, 3, 1, 1, 1)?;

        Ok(AcousticEncoderWeights {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

/// Load the acoustic (BigCodec) encoder from a checkpoint.
pub fn load_acoustic_encoder<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<AcousticEncoder<R>> {
    let mut loader = SafeTensorsLoader::open(path)?;
    let weights = AcousticEncoderLoader::<R> {
        loader: &mut loader,
        device,
        prefix: DEFAULT_ACOUSTIC_ENCODER_PREFIX.to_string(),
    }
    .build()?;
    AcousticEncoder::new(weights)
}

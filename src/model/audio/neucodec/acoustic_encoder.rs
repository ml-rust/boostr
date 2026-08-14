//! NeuCodec's acoustic encoder (BigCodec `CodecEncoder`): 16 kHz waveform ->
//! 50 Hz latent, downsampling by 320.
//!
//! Structure, verified against the checkpoint AND the upstream source (the HF
//! export renames the package's `conv_blocks` sequence):
//!
//! ```text
//! x [B, 1, T]
//!   -> conv1        Conv1d(1 -> 48, k=7, pad=3)
//!   -> block.0      EncoderBlock(48  -> 96,   stride 2)
//!   -> block.1      EncoderBlock(96  -> 192,  stride 2)
//!   -> block.2      EncoderBlock(192 -> 384,  stride 4)
//!   -> block.3      EncoderBlock(384 -> 768,  stride 4)
//!   -> block.4      EncoderBlock(768 -> 1536, stride 5)
//!   -> snake1       Activation1d(SnakeBeta(1536))
//!   -> conv2        Conv1d(1536 -> 1024, k=3, pad=1)
//!   [B, 1024, T/320]
//! ```
//!
//! Each `EncoderBlock(dim, stride)` is three `ResidualUnit`s at dilations
//! 1/3/9 on `dim/2` channels, then an `Activation1d`, then the strided
//! `Conv1d(dim/2 -> dim, k = 2*stride, stride, pad = stride/2 + stride%2)`.
//!
//! Each `ResidualUnit(dim, dilation)` is
//! `x + conv2(act(conv1(act(x))))` with `conv1` a dilated k=7 (padded to keep
//! length) and `conv2` a pointwise k=1.
//!
//! Upstream wraps every conv in `weight_norm`, but the HF export stores the
//! FUSED weights (plain `weight`, no `weight_g`/`weight_v`), so plain `Conv1d`
//! is correct here — no fusing step is needed at load time.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::alias_free::Activation1d;
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::Conv1d;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Downsampling strides per `EncoderBlock` (`up_ratios` upstream). Their
/// product, 320, is the 16 kHz -> 50 Hz ratio.
pub const ENCODER_STRIDES: [usize; 5] = [2, 2, 4, 4, 5];
/// Dilations of the three `ResidualUnit`s inside each block.
pub const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];
/// Base channel width (`ngf`); channels double at every block.
pub const ENCODER_BASE_CHANNELS: usize = 48;
/// Kernel size of the dilated conv inside a `ResidualUnit`.
pub const RESIDUAL_KERNEL_SIZE: usize = 7;

/// Total downsampling factor: the product of [`ENCODER_STRIDES`].
pub fn encoder_hop_length() -> usize {
    ENCODER_STRIDES.iter().product()
}

/// Padding that keeps length constant for a dilated odd kernel.
pub fn same_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size - 1) * dilation / 2
}

/// Padding upstream uses on the strided downsampling conv:
/// `stride / 2 + stride % 2`.
pub fn downsample_padding(stride: usize) -> usize {
    stride / 2 + stride % 2
}

/// `x + conv2(act2(conv1(act1(x))))`, length-preserving.
pub struct ResidualUnit<R: Runtime> {
    snake1: Activation1d<R>,
    conv1: Conv1d<R>,
    snake2: Activation1d<R>,
    conv2: Conv1d<R>,
}

/// Already-built weights for one [`ResidualUnit`].
pub struct ResidualUnitWeights<R: Runtime> {
    pub snake1: Activation1d<R>,
    pub conv1: Conv1d<R>,
    pub snake2: Activation1d<R>,
    pub conv2: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> ResidualUnit<R> {
    pub fn new(weights: ResidualUnitWeights<R>) -> Self {
        Self {
            snake1: weights.snake1,
            conv1: weights.conv1,
            snake2: weights.snake2,
            conv2: weights.conv2,
        }
    }

    /// `x [B, C, T] -> [B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let h = self.snake1.forward(client, x)?;
        let h = self.conv1.forward(client, &h)?;
        let h = self.snake2.forward(client, &h)?;
        let h = self.conv2.forward(client, &h)?;
        var_add(x, &h, client).map_err(Error::Numr)
    }
}

/// Three residual units, an alias-free activation, then a strided conv that
/// halves/quarters/fifths the time axis and doubles the channels.
pub struct EncoderBlock<R: Runtime> {
    res_units: Vec<ResidualUnit<R>>,
    snake1: Activation1d<R>,
    conv1: Conv1d<R>,
}

/// Already-built weights for one [`EncoderBlock`].
pub struct EncoderBlockWeights<R: Runtime> {
    pub res_units: Vec<ResidualUnit<R>>,
    pub snake1: Activation1d<R>,
    pub conv1: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> EncoderBlock<R> {
    pub fn new(weights: EncoderBlockWeights<R>) -> Result<Self> {
        if weights.res_units.len() != RESIDUAL_DILATIONS.len() {
            return Err(Error::InvalidArgument {
                arg: "res_units",
                reason: format!(
                    "expected {} residual units, got {}",
                    RESIDUAL_DILATIONS.len(),
                    weights.res_units.len()
                ),
            });
        }
        Ok(Self {
            res_units: weights.res_units,
            snake1: weights.snake1,
            conv1: weights.conv1,
        })
    }

    /// `x [B, C, T] -> [B, 2C, T/stride]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let mut h = x.alias();
        for unit in &self.res_units {
            h = unit.forward(client, &h)?;
        }
        let h = self.snake1.forward(client, &h)?;
        self.conv1.forward(client, &h)
    }
}

/// The full acoustic encoder.
pub struct AcousticEncoder<R: Runtime> {
    conv1: Conv1d<R>,
    blocks: Vec<EncoderBlock<R>>,
    snake1: Activation1d<R>,
    conv2: Conv1d<R>,
}

/// Already-built weights for [`AcousticEncoder`].
pub struct AcousticEncoderWeights<R: Runtime> {
    pub conv1: Conv1d<R>,
    pub blocks: Vec<EncoderBlock<R>>,
    pub snake1: Activation1d<R>,
    pub conv2: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> AcousticEncoder<R> {
    pub fn new(weights: AcousticEncoderWeights<R>) -> Result<Self> {
        if weights.blocks.len() != ENCODER_STRIDES.len() {
            return Err(Error::InvalidArgument {
                arg: "blocks",
                reason: format!(
                    "expected {} encoder blocks, got {}",
                    ENCODER_STRIDES.len(),
                    weights.blocks.len()
                ),
            });
        }
        Ok(Self {
            conv1: weights.conv1,
            blocks: weights.blocks,
            snake1: weights.snake1,
            conv2: weights.conv2,
        })
    }

    /// Forward: waveform `[B, 1, T]` -> latent `[B, 1024, T / 320]`.
    ///
    /// Returned CHANNELS-FIRST. Upstream's `CodecEncoder.forward` ends with a
    /// `permute(0, 2, 1)` and `encode_code` immediately transposes it back, so
    /// the two cancel; this port skips both.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != 1 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected mono waveform [B, 1, T], got {shape:?}"),
            });
        }

        let mut h = self.conv1.forward(client, x)?;
        for block in &self.blocks {
            h = block.forward(client, &h)?;
        }
        let h = self.snake1.forward(client, &h)?;
        self.conv2.forward(client, &h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_length_is_320() {
        assert_eq!(encoder_hop_length(), 320);
        // 16 kHz / 320 = the documented 50 Hz token rate.
        assert_eq!(16_000 / encoder_hop_length(), 50);
    }

    /// The dilated k=7 convs must preserve length, or the residual add fails.
    #[test]
    fn residual_padding_preserves_length() {
        for d in RESIDUAL_DILATIONS {
            let pad = same_padding(RESIDUAL_KERNEL_SIZE, d);
            let span = (RESIDUAL_KERNEL_SIZE - 1) * d + 1;
            let len_in = 128usize;
            let len_out = len_in + 2 * pad - span + 1;
            assert_eq!(len_out, len_in, "dilation {d} must preserve length");
        }
    }

    /// Upstream's `stride/2 + stride%2` padding with `k = 2*stride` divides the
    /// length by exactly `stride`.
    #[test]
    fn downsample_padding_divides_length_by_stride() {
        for stride in ENCODER_STRIDES {
            let pad = downsample_padding(stride);
            let k = 2 * stride;
            let len_in = 320usize;
            let len_out = (len_in + 2 * pad - k) / stride + 1;
            assert_eq!(
                len_out,
                len_in / stride,
                "stride {stride} (pad {pad}, k {k}) must divide length exactly"
            );
        }
    }
}

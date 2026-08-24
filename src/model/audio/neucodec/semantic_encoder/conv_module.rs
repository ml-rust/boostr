//! `Wav2Vec2BertConvolutionModule` — the conformer layer's convolution branch.
//!
//! ```text
//! x [B, T, C]
//!   -> layer_norm                         # internal; the LAYER adds no pre-norm
//!   -> [B, C, T]
//!   -> pointwise_conv1   (C -> 2C, k=1, no bias)
//!   -> GLU over CHANNELS: a * sigmoid(b), a = first C, b = last C
//!   -> pad (kernel-1) on the LEFT only    # CAUSAL
//!   -> depthwise_conv    (C -> C, k=31, groups=C, no bias, VALID)
//!   -> depthwise_layer_norm               # AFTER the conv, BEFORE the activation
//!   -> SiLU
//!   -> pointwise_conv2   (C -> C, k=1, no bias)
//!   -> [B, T, C]
//! ```
//!
//! The module applies NO residual of its own — the enclosing layer owns it.
//!
//! ## The depthwise padding is CAUSAL: all 30 on the left, none on the right
//!
//! The plausible-but-wrong alternative is the symmetric "same" padding every
//! other conv in this codec uses (`(k-1)/2 = 15` on each side), which also
//! preserves the length and so passes every shape test. It is a different
//! model: this branch is streaming, so each frame may only see history. With
//! symmetric padding every output frame leaks 15 frames of future context, and
//! nothing in the tensor shapes ever complains.
//!
//! ## GLU splits the CHANNEL axis, and `sigmoid` gates the SECOND half
//!
//! `pointwise_conv1` doubles the channels precisely so the GLU can halve them
//! again. Splitting the time axis instead would halve `T` and desynchronize the
//! branch from the residual. Within the split, the gate is the second half:
//! `out = a * sigmoid(b)`. Swapping to `sigmoid(a) * b` is shape-identical and
//! silently wrong.
//!
//! ## `depthwise_layer_norm` sits between the conv and the activation
//!
//! It is a genuine LayerNorm over channels applied in `[B, T, C]` layout, so
//! the tensor is transposed out of and back into channels-first around it. The
//! natural-looking placements — before the depthwise conv (where the module's
//! own `layer_norm` already is) or after the SiLU — both typecheck and both
//! change the output.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::{Conv1d, LayerNorm, var_contiguous};
use numr::autograd::{Var, var_narrow, var_permute, var_sigmoid_mul, var_silu};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Already-built weights for [`ConvolutionModule`].
pub struct ConvolutionModuleWeights<R: Runtime> {
    /// `conv_module.layer_norm.{weight,bias}`, width = hidden.
    pub layer_norm: LayerNorm<R>,
    /// `conv_module.pointwise_conv1.weight`, `[2C, C, 1]`, no bias.
    pub pointwise_conv1: Conv1d<R>,
    /// `conv_module.depthwise_conv.weight`, `[C, 1, 31]`, groups = C, no bias.
    /// Must already carry the causal left-only padding.
    pub depthwise_conv: Conv1d<R>,
    /// `conv_module.depthwise_layer_norm.{weight,bias}`, width = hidden.
    pub depthwise_layer_norm: LayerNorm<R>,
    /// `conv_module.pointwise_conv2.weight`, `[C, C, 1]`, no bias.
    pub pointwise_conv2: Conv1d<R>,
}

/// Conformer convolution module: `[B, T, C] -> [B, T, C]`, residual-free.
pub struct ConvolutionModule<R: Runtime> {
    layer_norm: LayerNorm<R>,
    pointwise_conv1: Conv1d<R>,
    depthwise_conv: Conv1d<R>,
    depthwise_layer_norm: LayerNorm<R>,
    pointwise_conv2: Conv1d<R>,
    channels: usize,
}

impl<R: Runtime> ConvolutionModule<R> {
    /// Assemble from already-loaded weights. `channels` is the residual width.
    pub fn new(weights: ConvolutionModuleWeights<R>, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(Error::InvalidArgument {
                arg: "channels",
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            layer_norm: weights.layer_norm,
            pointwise_conv1: weights.pointwise_conv1,
            depthwise_conv: weights.depthwise_conv,
            depthwise_layer_norm: weights.depthwise_layer_norm,
            pointwise_conv2: weights.pointwise_conv2,
            channels,
        })
    }
}

impl<R: Runtime<DType = DType>> ConvolutionModule<R> {
    /// Forward: `x [B, T, C] -> [B, T, C]`. Length-preserving.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.channels {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, T, {}], got {shape:?}", self.channels),
            });
        }

        let h = self.layer_norm.forward(client, x)?;
        let h = swap_time_and_channels(&h)?;

        let h = self.pointwise_conv1.forward(client, &h)?;
        let h = self.glu(client, &h)?;

        // Causal: the left-only pad lives in the Conv1d's PaddingMode.
        let h = self.depthwise_conv.forward(client, &h)?;

        // Norm AFTER the conv and BEFORE the activation, in [B, T, C] layout.
        let h = swap_time_and_channels(&h)?;
        let h = self.depthwise_layer_norm.forward(client, &h)?;
        let h = swap_time_and_channels(&h)?;

        let h = var_silu(&h, client).map_err(Error::Numr)?;
        let h = self.pointwise_conv2.forward(client, &h)?;
        swap_time_and_channels(&h)
    }

    /// Gated linear unit over the channel axis: `a * sigmoid(b)`.
    fn glu<C>(&self, client: &C, h: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = h.shape();
        if shape.len() != 3 || shape[1] != 2 * self.channels {
            return Err(Error::ModelError {
                reason: format!(
                    "pointwise_conv1 must produce [B, {}, T], got {shape:?}",
                    2 * self.channels
                ),
            });
        }
        let value = var_narrow(h, 1, 0, self.channels).map_err(Error::Numr)?;
        let gate = var_narrow(h, 1, self.channels, self.channels).map_err(Error::Numr)?;
        let value = var_contiguous(&value)?;
        let gate = var_contiguous(&gate)?;

        // `var_sigmoid_mul(a, b) = sigmoid(a) * b`, so the GATE goes first to
        // get `value * sigmoid(gate)`.
        var_sigmoid_mul(&gate, &value, client).map_err(Error::Numr)
    }
}

/// Swap the last two axes of a rank-3 `Var`, materializing the result.
///
/// `[B, T, C] <-> [B, C, T]`. Self-inverse, so one helper covers both
/// directions.
fn swap_time_and_channels<R: Runtime>(x: &Var<R>) -> Result<Var<R>> {
    let permuted = var_permute(x, &[0, 2, 1]).map_err(Error::Numr)?;
    var_contiguous(&permuted)
}

/// Left-only ("causal") padding for a depthwise conv of the given kernel size.
///
/// `kernel_size - 1` frames on the left and zero on the right, so output frame
/// `t` depends on input frames `t - (k - 1) ..= t` and never on the future.
pub fn causal_padding(kernel_size: usize) -> numr::ops::PaddingMode {
    numr::ops::PaddingMode::conv1d(kernel_size.saturating_sub(1), 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn layer_norm(dim: usize, device: &CpuDevice) -> LayerNorm<CpuRuntime> {
        LayerNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; dim], &[dim], device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; dim], &[dim], device).unwrap(),
            1e-5,
            false,
        )
    }

    fn conv(
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        groups: usize,
        padding: PaddingMode,
        device: &CpuDevice,
    ) -> Conv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.05f32; out_ch * in_ch * kernel],
            &[out_ch, in_ch, kernel],
            device,
        )
        .unwrap();
        Conv1d::new(weight, None, 1, padding, 1, groups, false)
    }

    fn module(channels: usize, kernel: usize, device: &CpuDevice) -> ConvolutionModule<CpuRuntime> {
        ConvolutionModule::new(
            ConvolutionModuleWeights {
                layer_norm: layer_norm(channels, device),
                pointwise_conv1: conv(
                    2 * channels,
                    channels,
                    1,
                    1,
                    PaddingMode::Custom(0, 0, 0, 0),
                    device,
                ),
                depthwise_conv: conv(
                    channels,
                    1,
                    kernel,
                    channels,
                    causal_padding(kernel),
                    device,
                ),
                depthwise_layer_norm: layer_norm(channels, device),
                pointwise_conv2: conv(
                    channels,
                    channels,
                    1,
                    1,
                    PaddingMode::Custom(0, 0, 0, 0),
                    device,
                ),
            },
            channels,
        )
        .expect("build conv module")
    }

    #[test]
    fn causal_padding_is_left_only() {
        assert_eq!(causal_padding(31), PaddingMode::Custom(30, 0, 0, 0));
        assert_eq!(causal_padding(1), PaddingMode::Custom(0, 0, 0, 0));
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let (channels, kernel, t) = (8, 5, 12);
        let m = module(channels, kernel, &device);

        let data: Vec<f32> = (0..(t * channels))
            .map(|i| (i as f32 * 0.09).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, t, channels], &device).unwrap(),
            false,
        );
        let y = m.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[1, t, channels]);
        for v in y.tensor().contiguous().expect("contiguous").to_vec::<f32>() {
            assert!(v.is_finite(), "conv module output is not finite: {v}");
        }
    }

    #[test]
    fn rejects_wrong_channel_width() {
        let (client, device) = cpu_setup();
        let m = module(8, 5, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4 * 3], &[1, 4, 3], &device).unwrap(),
            false,
        );
        assert!(m.forward(&client, &x).is_err());
    }
}

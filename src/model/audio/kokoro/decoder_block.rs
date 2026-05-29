//! Decoder building blocks for Kokoro / StyleTTS2 ISTFTNet generator.
//!
//! Two small, reusable units:
//!
//! * [`DecoderBlock`] — the AdaIN-conditioned residual Conv1d block used in
//!   every decoder stage.
//! * [`UpsampleBlock`] — transposed-conv upsampling with style-conditioned
//!   activation, used between residual stacks.
//!
//! Higher-level assembly (number of stages, upsample ratios, residual counts,
//! magnitude / phase heads) lives with the Kokoro loader in M7, where the
//! concrete checkpoint pins every shape. These blocks are the atoms.
//!
//! Reference flow per block (AdaIN-Conv residual, pre-activation variant):
//!
//! ```text
//!     y = x + conv2(leaky_relu(adain2(conv1(leaky_relu(adain1(x, γ1, β1))), γ2, β2)))
//! ```
//!
//! Inference-only; no autograd.

use crate::error::{Error, Result};
use crate::nn::{AdaIn1d, Conv1d};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ConvOps, NormalizationOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// AdaIN-conditioned residual Conv1d block.
pub struct DecoderBlock<R: Runtime> {
    adain1: AdaIn1d,
    adain2: AdaIn1d,
    conv1: Conv1d<R>,
    conv2: Conv1d<R>,
    leaky_slope: f64,
}

impl<R: Runtime> DecoderBlock<R> {
    pub fn new(
        adain1: AdaIn1d,
        adain2: AdaIn1d,
        conv1: Conv1d<R>,
        conv2: Conv1d<R>,
        leaky_slope: f64,
    ) -> Result<Self> {
        if adain1.channels() != adain2.channels() {
            return Err(Error::InvalidArgument {
                arg: "adain2",
                reason: "AdaIN1d pair must share channel count".into(),
            });
        }
        Ok(Self {
            adain1,
            adain2,
            conv1,
            conv2,
            leaky_slope,
        })
    }

    pub fn channels(&self) -> usize {
        self.adain1.channels()
    }

    /// Forward: `[B, C, T]` + `(gamma1, beta1), (gamma2, beta2)` each `[B, C]`.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        gamma1: &Tensor<R>,
        beta1: &Tensor<R>,
        gamma2: &Tensor<R>,
        beta2: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + BinaryOps<R>
            + UtilityOps<R>
            + ActivationOps<R>
            + TensorOps<R>,
    {
        let h1 = self.adain1.forward(client, x, gamma1, beta1)?;
        let h1 = client
            .leaky_relu(&h1, self.leaky_slope)
            .map_err(Error::Numr)?;
        let h1 = self.conv1.forward_inference(client, &h1)?;

        let h2 = self.adain2.forward(client, &h1, gamma2, beta2)?;
        let h2 = client
            .leaky_relu(&h2, self.leaky_slope)
            .map_err(Error::Numr)?;
        let h2 = self.conv2.forward_inference(client, &h2)?;

        // Residual skip. Shapes of `x` and `h2` must match at the time-axis
        // (the conv pair is same-padded in Kokoro); callers that stride inside
        // a residual block must match the skip path by stacking a stride-1
        // 1×1 conv on `x` before passing it here.
        client.add(x, &h2).map_err(Error::Numr)
    }
}

/// Upsample stage: transposed Conv1d + optional leaky-ReLU activation.
///
/// Kokoro's generator typically alternates between `UpsampleBlock`s
/// (doubling the time axis) and stacks of `DecoderBlock`s. This wrapper keeps
/// the upsample config explicit (stride, padding, output_padding) so callers
/// can line it up with the checkpoint's recipe.
pub struct UpsampleBlock<R: Runtime> {
    weight: Tensor<R>,
    bias: Option<Tensor<R>>,
    stride: usize,
    padding: numr::ops::PaddingMode,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    leaky_slope: f64,
}

impl<R: Runtime> UpsampleBlock<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        stride: usize,
        padding: numr::ops::PaddingMode,
        output_padding: usize,
        dilation: usize,
        groups: usize,
        leaky_slope: f64,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            leaky_slope,
        }
    }

    /// Forward: `[B, C_in, T]` → `[B, C_out, T * stride]` (approximately).
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + ActivationOps<R>,
    {
        let up = client
            .conv_transpose1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            )
            .map_err(Error::Numr)?;
        client
            .leaky_relu(&up, self.leaky_slope)
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    fn build_block(device: &<CpuRuntime as Runtime>::Device) -> DecoderBlock<CpuRuntime> {
        let c = 4;
        let k = 3;
        let conv = || {
            Conv1d::new(
                zeros(&[c, c, k], device),
                Some(zeros(&[c], device)),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            )
        };
        DecoderBlock::new(
            AdaIn1d::new(c, 1e-5),
            AdaIn1d::new(c, 1e-5),
            conv(),
            conv(),
            0.2,
        )
        .unwrap()
    }

    #[test]
    fn decoder_block_preserves_shape() {
        let (client, device) = cpu_setup();
        let block = build_block(&device);
        let c = block.channels();
        let x = zeros(&[1, c, 6], &device);
        let gamma = zeros(&[1, c], &device);
        let beta = zeros(&[1, c], &device);
        let y = block
            .forward(&client, &x, &gamma, &beta, &gamma, &beta)
            .unwrap();
        assert_eq!(y.shape(), &[1, c, 6]);
    }

    #[test]
    fn decoder_block_zero_weights_is_identity() {
        // With all-zero conv weights/biases and unit gamma / zero beta, the
        // residual path dominates: y = x + 0 = x.
        let (client, device) = cpu_setup();
        let c = 2;
        let k = 3;
        let zero_conv = || {
            Conv1d::new(
                zeros(&[c, c, k], &device),
                Some(zeros(&[c], &device)),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            )
        };
        let block = DecoderBlock::new(
            AdaIn1d::new(c, 1e-5),
            AdaIn1d::new(c, 1e-5),
            zero_conv(),
            zero_conv(),
            0.2,
        )
        .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            &device,
        );
        let gamma = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2], &device);
        let beta = zeros(&[1, 2], &device);
        let y = block
            .forward(&client, &x, &gamma, &beta, &gamma, &beta)
            .unwrap();
        let a: Vec<f32> = x.to_vec();
        let b: Vec<f32> = y.to_vec();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-5, "{u} vs {v}");
        }
    }

    #[test]
    fn decoder_block_rejects_channel_mismatch() {
        let adain1 = AdaIn1d::new(4, 1e-5);
        let adain2 = AdaIn1d::new(5, 1e-5);
        let (_client, device) = cpu_setup();
        let conv = || {
            Conv1d::new(
                zeros(&[4, 4, 3], &device),
                None,
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            )
        };
        assert!(DecoderBlock::new(adain1, adain2, conv(), conv(), 0.2).is_err());
    }

    #[test]
    fn upsample_block_doubles_time_axis() {
        let (client, device) = cpu_setup();
        // C_in = 1, C_out = 1, kernel = 2, stride = 2 → output ~ T * 2.
        let block = UpsampleBlock::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 1, 2], &device),
            None,
            2,
            PaddingMode::Valid,
            0,
            1,
            1,
            0.0,
        );
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device);
        let y = block.forward(&client, &x).unwrap();
        // L_out = (3-1)*2 + 2 = 6.
        assert_eq!(y.shape(), &[1, 1, 6]);
    }
}

//! `AdainResBlk1d` — single-path AdaIN residual block.
//!
//! Different module from [`crate::model::audio::kokoro::AdaINResBlock1`]
//! despite the name similarity:
//!
//! * **`AdainResBlk1d`** (this file): one `conv1` + one `conv2`, optional
//!   `conv1x1` shortcut for channel-count change, optional `pool`
//!   (ConvTranspose1d) for upsampling. Used in `predictor.F0.*`,
//!   `predictor.N.*`, `decoder.encode`, `decoder.decode.*`. No Snake —
//!   uses LeakyReLU.
//! * **`AdaINResBlock1`** (generator.resblocks): three tiers of
//!   `(AdaIN → Snake → Conv1d) × 2` with residual skip per tier. Used
//!   only in `decoder.generator.resblocks`.
//!
//! Upstream forward:
//! ```text
//! shortcut = conv1x1(x) if learned_sc else x
//! if upsample:
//!     x_pool = pool(x)
//!     shortcut = pool(shortcut)  # also pooled
//! xt = adain1(x, s)
//! xt = leaky_relu(xt, 0.2)
//! xt = pool(xt) if upsample else xt
//! xt = conv1(xt)
//! xt = adain2(xt, s)
//! xt = leaky_relu(xt, 0.2)
//! xt = conv2(xt)
//! return (xt + shortcut) / sqrt(2)
//! ```

use crate::error::{Error, Result};
use crate::model::audio::kokoro::KokoroAdaIn1d;
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, PaddingMode, ScalarOps,
    TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Nearest-neighbor upsample by `scale` along the last (time) axis of a
/// `[B, C, T]` tensor. Output length is `T * scale`. Matches
/// `nn.Upsample(scale_factor=scale, mode='nearest')` on rank-3 inputs.
///
/// Implemented via `narrow` + `cat`: for each output position `t`, reads
/// input position `t / scale`. Kept backend-agnostic so the shortcut path
/// doesn't force CPU specialization.
fn nearest_upsample_1d<R: Runtime<DType = DType>>(
    x: &Tensor<R>,
    scale: usize,
) -> Result<Tensor<R>> {
    if scale == 0 {
        return Err(Error::InvalidArgument {
            arg: "scale",
            reason: "must be > 0".into(),
        });
    }
    if scale == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [B, C, T], got {shape:?}"),
        });
    }
    let t = shape[2];

    // Reshape to [B, C, T, 1], broadcast_to [B, C, T, scale], then merge the
    // last two dims by reshape back to [B, C, T*scale]. Broadcasting a
    // stride-0 view over a new trailing axis is a zero-copy duplicate; the
    // final reshape materializes it once via contiguous().
    let unsq = x
        .reshape(&[shape[0], shape[1], t, 1])
        .map_err(Error::Numr)?;
    let broadcast = unsq
        .broadcast_to(&[shape[0], shape[1], t, scale])
        .map_err(Error::Numr)?
        .contiguous()?;
    broadcast
        .reshape(&[shape[0], shape[1], t * scale])
        .map_err(Error::Numr)
}

/// Single-path AdaIN residual block.
pub struct AdainResBlk1d<R: Runtime> {
    adain1: KokoroAdaIn1d<R>,
    adain2: KokoroAdaIn1d<R>,
    conv1: Conv1d<R>,
    conv2: Conv1d<R>,
    /// Optional 1×1 conv shortcut when `dim_in != dim_out`. If `None`, the
    /// residual is applied to `x` directly.
    conv1x1: Option<Conv1d<R>>,
    /// Optional upsampling transposed-conv (scale factor = 2 in upstream).
    /// Pool weight stored as a ConvTranspose1d weight layout `[C_in, C_out, K]`.
    pool: Option<PoolParams<R>>,
    leaky_slope: f64,
}

/// Parameters for the optional upsampling ConvTranspose1d step.
pub struct PoolParams<R: Runtime> {
    pub weight: Tensor<R>,
    pub bias: Option<Tensor<R>>,
    pub stride: usize,
    pub padding: PaddingMode,
    pub output_padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl<R: Runtime> AdainResBlk1d<R> {
    pub fn new(
        adain1: KokoroAdaIn1d<R>,
        adain2: KokoroAdaIn1d<R>,
        conv1: Conv1d<R>,
        conv2: Conv1d<R>,
        conv1x1: Option<Conv1d<R>>,
        pool: Option<PoolParams<R>>,
        leaky_slope: f64,
    ) -> Self {
        Self {
            adain1,
            adain2,
            conv1,
            conv2,
            conv1x1,
            pool,
            leaky_slope,
        }
    }

    /// Forward: `x [B, C_in, T_in]`, `style [B, style_dim]` → `[B, C_out, T_out]`.
    ///
    /// `T_out = T_in` when `pool` is `None`; otherwise determined by the pool's
    /// transpose-conv output formula.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + UtilityOps<R>,
    {
        // Shortcut path: 1×1 conv (if learned_sc), then weightless nearest
        // upsample (if the main path upsamples). Upstream uses
        // `nn.UpSample1d(scale=2)` here — NOT the depthwise `pool` used on
        // the main path. Matching main-path output length is what matters.
        let mut shortcut = match &self.conv1x1 {
            Some(c) => c.forward_inference(client, x)?,
            None => x.clone(),
        };
        if let Some(p) = &self.pool {
            shortcut = nearest_upsample_1d(&shortcut, p.stride)?;
        }

        // Main path.
        let mut h = self.adain1.forward(client, x, style)?;
        h = client
            .leaky_relu(&h, self.leaky_slope)
            .map_err(Error::Numr)?;
        if let Some(p) = &self.pool {
            h = client
                .conv_transpose1d(
                    &h,
                    &p.weight,
                    p.bias.as_ref(),
                    p.stride,
                    p.padding,
                    p.output_padding,
                    p.dilation,
                    p.groups,
                )
                .map_err(Error::Numr)?;
        }
        h = self.conv1.forward_inference(client, &h)?;
        h = self.adain2.forward(client, &h, style)?;
        h = client
            .leaky_relu(&h, self.leaky_slope)
            .map_err(Error::Numr)?;
        h = self.conv2.forward_inference(client, &h)?;

        // (xt + shortcut) / sqrt(2)
        let sum = client.add(&h, &shortcut).map_err(Error::Numr)?;
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        client.mul_scalar(&sum, inv_sqrt2).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }
    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }

    fn build_adain(
        channels: usize,
        style_dim: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> KokoroAdaIn1d<CpuRuntime> {
        KokoroAdaIn1d::new(
            zeros(&[2 * channels, style_dim], device),
            zeros(&[2 * channels], device),
            ones(&[channels], device),
            zeros(&[channels], device),
            1e-5,
        )
        .unwrap()
    }

    fn conv(
        c_out: usize,
        c_in: usize,
        k: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            zeros(&[c_out, c_in, k], device),
            Some(zeros(&[c_out], device)),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    #[test]
    fn no_upsample_no_shortcut_preserves_shape() {
        let (client, device) = cpu_setup();
        let block = AdainResBlk1d::new(
            build_adain(4, 2, &device),
            build_adain(4, 2, &device),
            conv(4, 4, 3, &device),
            conv(4, 4, 3, &device),
            None,
            None,
            0.2,
        );
        let x = zeros(&[1, 4, 8], &device);
        let style = zeros(&[1, 2], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8]);
    }

    #[test]
    fn zero_everything_gives_zero_output() {
        let (client, device) = cpu_setup();
        let block = AdainResBlk1d::new(
            build_adain(2, 2, &device),
            build_adain(2, 2, &device),
            conv(2, 2, 3, &device),
            conv(2, 2, 3, &device),
            None,
            None,
            0.2,
        );
        let x = zeros(&[1, 2, 4], &device);
        let style = zeros(&[1, 2], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        for v in y.to_vec::<f32>() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn with_conv1x1_shortcut_changes_channels() {
        let (client, device) = cpu_setup();
        // dim_in=4 -> dim_out=6 via 1x1 conv shortcut. Main path conv1 must map 4->6.
        let block = AdainResBlk1d::new(
            build_adain(4, 2, &device),
            build_adain(6, 2, &device),
            conv(6, 4, 3, &device),
            conv(6, 6, 3, &device),
            Some(conv(6, 4, 1, &device)),
            None,
            0.2,
        );
        let x = zeros(&[1, 4, 8], &device);
        let style = zeros(&[1, 2], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        assert_eq!(y.shape(), &[1, 6, 8]);
    }

    #[test]
    fn with_pool_doubles_time_axis() {
        let (client, device) = cpu_setup();
        // Pool = ConvTranspose1d(4, 4, kernel=2, stride=2) → T doubles.
        let pool = PoolParams {
            weight: zeros(&[4, 4, 2], &device),
            bias: None,
            stride: 2,
            padding: PaddingMode::Valid,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        };
        let block = AdainResBlk1d::new(
            build_adain(4, 2, &device),
            build_adain(4, 2, &device),
            conv(4, 4, 3, &device),
            conv(4, 4, 3, &device),
            None,
            Some(pool),
            0.2,
        );
        let x = zeros(&[1, 4, 4], &device);
        let style = zeros(&[1, 2], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        // L_out for ConvTranspose1d(k=2, s=2) on L=4: (4-1)*2 + 2 = 8.
        assert_eq!(y.shape(), &[1, 4, 8]);
    }
}

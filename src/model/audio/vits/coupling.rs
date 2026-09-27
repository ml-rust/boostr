//! `VitsResidualCouplingBlock` / `VitsResidualCouplingLayer`, reverse
//! (inference) path only, plus the prior-latent sampling step that feeds
//! it.
//!
//! Exact port of HF
//! `transformers.models.vits.modeling_vits.VitsResidualCouplingBlock` and
//! `VitsResidualCouplingLayer`. HF's layer always runs mean-only (it sets
//! `log_stddev = torch.zeros_like(mean)` unconditionally — there is no
//! config flag that changes this), so the reverse transform simplifies to
//! `(second_half - mean) * padding_mask`; there is no `exp(-log_stddev)`
//! term to carry since it is always `1`. The training-time forward branch
//! (`reverse=False`) is not ported: this crate is inference-only.

use crate::error::{Error, Result};
use crate::model::audio::vits::wavenet::VitsWaveNet;
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, ConvOps, PaddingMode, RandomOps, ReduceOps, ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One coupling layer: splits `[batch, flow_size, length]` into two equal
/// halves, runs a `VitsWaveNet` conditioned on the first half to predict a
/// per-position mean for the second half, and (in reverse) subtracts it
/// back out.
pub struct VitsResidualCouplingLayer<R: Runtime> {
    conv_pre: Conv1d<R>,
    wavenet: VitsWaveNet<R>,
    conv_post: Conv1d<R>,
    half_channels: usize,
}

impl<R: Runtime<DType = DType>> VitsResidualCouplingLayer<R> {
    /// Reads `conv_pre.*`, `wavenet.*`, `conv_post.*` under `vb`'s current
    /// prefix. `conv_pre`/`conv_post` are plain (not weight-norm
    /// reparameterized) `nn.Conv1d` in HF — only the WaveNet's internal
    /// convs carry weight-norm.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &VitsConfig,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        let half_channels = config.flow_size / 2;

        let mut pre_vb = vb.pp("conv_pre");
        let pre_weight = pre_vb.take_tensor("weight")?;
        let pre_bias = pre_vb.take_tensor("bias")?;
        drop(pre_vb);
        let conv_pre = Conv1d::new(
            pre_weight,
            Some(pre_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        let mut wn_vb = vb.pp("wavenet");
        let wavenet = VitsWaveNet::from_varbuilder(
            client,
            &mut wn_vb,
            config.hidden_size,
            config.wavenet_kernel_size,
            config.wavenet_dilation_rate,
            config.prior_encoder_num_wavenet_layers,
            config.speaker_embedding_size,
        )?;
        drop(wn_vb);

        let mut post_vb = vb.pp("conv_post");
        let post_weight = post_vb.take_tensor("weight")?;
        let post_bias = post_vb.take_tensor("bias")?;
        drop(post_vb);
        let conv_post = Conv1d::new(
            post_weight,
            Some(post_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        Ok(Self {
            conv_pre,
            wavenet,
            conv_post,
            half_channels,
        })
    }

    /// `inputs`: `[batch, flow_size, length]`. `padding_mask`: `[batch, 1,
    /// length]`. `global_conditioning`: `[batch, speaker_embedding_size,
    /// 1]`, required iff the loaded `WaveNet` has a `cond_layer`.
    pub fn reverse<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let first_half = inputs
            .narrow(1, 0, self.half_channels)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let second_half = inputs
            .narrow(1, self.half_channels, self.half_channels)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        let hidden = self.conv_pre.forward_inference(client, &first_half)?;
        let hidden = client.mul(&hidden, padding_mask).map_err(Error::Numr)?;
        let hidden = self
            .wavenet
            .forward(client, &hidden, padding_mask, global_conditioning)?;
        let mean = self.conv_post.forward_inference(client, &hidden)?;
        let mean = client.mul(&mean, padding_mask).map_err(Error::Numr)?;

        // reverse: second_half = (second_half - mean) * exp(-log_stddev) *
        // padding_mask, and log_stddev is always zero in HF (see module
        // doc), so exp(-log_stddev) == 1.
        let diff = client.sub(&second_half, &mean).map_err(Error::Numr)?;
        let second_half = client.mul(&diff, padding_mask).map_err(Error::Numr)?;

        client
            .cat(&[&first_half, &second_half], 1)
            .map_err(Error::Numr)
    }
}

/// A stack of [`VitsResidualCouplingLayer`]s, run in reverse. HF's forward
/// path runs `flows` in order and flips after each one; the reverse path
/// (the only one this crate implements) runs `flows` in reverse order and
/// flips BEFORE each one — see `VitsResidualCouplingBlock.forward`'s
/// `reverse` branch.
pub struct VitsResidualCouplingBlock<R: Runtime> {
    flows: Vec<VitsResidualCouplingLayer<R>>,
}

impl<R: Runtime<DType = DType>> VitsResidualCouplingBlock<R> {
    /// Reads `flows.{0..prior_encoder_num_flows}.*` under `vb`'s current
    /// prefix (the checkpoint's `flow.` for the real MMS-TTS weights).
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &VitsConfig,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        let mut flows = Vec::with_capacity(config.prior_encoder_num_flows);
        for i in 0..config.prior_encoder_num_flows {
            let mut flow_vb = vb.pp(&format!("flows.{i}"));
            flows.push(VitsResidualCouplingLayer::from_varbuilder(
                client,
                &mut flow_vb,
                config,
            )?);
            drop(flow_vb);
        }
        Ok(Self { flows })
    }

    /// `inputs`: `[batch, flow_size, length]` (the sampled prior latents).
    /// `padding_mask`: `[batch, 1, length]`.
    pub fn reverse<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let mut inputs = inputs.clone();
        for flow in self.flows.iter().rev() {
            inputs = inputs
                .flip(1)
                .and_then(|t| t.contiguous())
                .map_err(Error::Numr)?;
            inputs = flow.reverse(client, &inputs, padding_mask, global_conditioning)?;
        }
        Ok(inputs)
    }
}

/// Draws standard-normal noise via [`RandomOps::randn_seeded`] and runs
/// [`sample_prior_latents_with_noise`]. The normal inference entry point;
/// `expanded_prior_means`/`expanded_prior_log_variances`:
/// `[batch, flow_size, length]` (HF's `regulate_length`-expanded prior),
/// `seed` picks the deterministic PRNG stream (see
/// [`RandomOps::randn_seeded`]).
pub fn sample_prior_latents<R, C>(
    client: &C,
    expanded_prior_means: &Tensor<R>,
    expanded_prior_log_variances: &Tensor<R>,
    noise_scale: f64,
    seed: u64,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RandomOps<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
{
    let noise = client
        .randn_seeded(
            expanded_prior_means.shape(),
            expanded_prior_means.dtype(),
            seed,
        )
        .map_err(Error::Numr)?;
    sample_prior_latents_with_noise(
        client,
        expanded_prior_means,
        expanded_prior_log_variances,
        noise_scale,
        &noise,
    )
}

/// Prior-latent sampling with a caller-supplied noise draw in place of a
/// fresh [`RandomOps::randn_seeded`] draw: `expanded_prior_means + noise *
/// exp(expanded_prior_log_variances) * noise_scale` (HF's
/// `prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances)
/// * self.noise_scale`). Exists so a parity test can replay a captured
/// `torch.randn_like(prior_means)` draw bit-for-bit instead of a
/// reproduced-but-necessarily-different RNG stream;
/// [`sample_prior_latents`] is the normal entry point.
pub fn sample_prior_latents_with_noise<R, C>(
    client: &C,
    expanded_prior_means: &Tensor<R>,
    expanded_prior_log_variances: &Tensor<R>,
    noise_scale: f64,
    noise: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
{
    let stddev = client
        .exp(expanded_prior_log_variances)
        .map_err(Error::Numr)?;
    let scaled_noise = client.mul(noise, &stddev).map_err(Error::Numr)?;
    let scaled_noise = client
        .mul_scalar(&scaled_noise, noise_scale)
        .map_err(Error::Numr)?;
    client
        .add(expanded_prior_means, &scaled_noise)
        .map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn tiny_config() -> VitsConfig {
        VitsConfig {
            flow_size: 4,
            hidden_size: 4,
            wavenet_kernel_size: 3,
            wavenet_dilation_rate: 1,
            prior_encoder_num_wavenet_layers: 2,
            prior_encoder_num_flows: 2,
            speaker_embedding_size: 0,
            ..Default::default()
        }
    }

    /// Inserts one [`VitsResidualCouplingLayer`]'s tensors into `map`, all
    /// named under `prefix` (empty for a standalone layer, `"flows.{i}."`
    /// for one entry of a [`VitsResidualCouplingBlock`]).
    fn insert_layer_map(
        map: &mut VarMap<CpuRuntime>,
        prefix: &str,
        cfg: &VitsConfig,
        device: &numr::runtime::cpu::CpuDevice,
    ) {
        let half = cfg.flow_size / 2;
        let h = cfg.hidden_size;
        map.insert(
            format!("{prefix}conv_pre.weight"),
            Tensor::<CpuRuntime>::from_slice(&vec![0.05f32; h * half], &[h, half, 1], device)
                .unwrap(),
        );
        map.insert(
            format!("{prefix}conv_pre.bias"),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; h], &[h], device).unwrap(),
        );
        for i in 0..cfg.prior_encoder_num_wavenet_layers {
            let k = cfg.wavenet_kernel_size;
            map.insert(
                format!("{prefix}wavenet.in_layers.{i}.weight"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.1f32; 2 * h * h * k],
                    &[2 * h, h, k],
                    device,
                )
                .unwrap(),
            );
            map.insert(
                format!("{prefix}wavenet.in_layers.{i}.bias"),
                Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; 2 * h], &[2 * h], device).unwrap(),
            );
            let res_skip_out = if i < cfg.prior_encoder_num_wavenet_layers - 1 {
                2 * h
            } else {
                h
            };
            map.insert(
                format!("{prefix}wavenet.res_skip_layers.{i}.weight"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.1f32; res_skip_out * h],
                    &[res_skip_out, h, 1],
                    device,
                )
                .unwrap(),
            );
            map.insert(
                format!("{prefix}wavenet.res_skip_layers.{i}.bias"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.0f32; res_skip_out],
                    &[res_skip_out],
                    device,
                )
                .unwrap(),
            );
        }
        map.insert(
            format!("{prefix}conv_post.weight"),
            Tensor::<CpuRuntime>::from_slice(&vec![0.05f32; half * h], &[half, h, 1], device)
                .unwrap(),
        );
        map.insert(
            format!("{prefix}conv_post.bias"),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; half], &[half], device).unwrap(),
        );
    }

    #[test]
    fn coupling_layer_reverse_masks_padding_and_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = VarMap::<CpuRuntime>::new();
        insert_layer_map(&mut map, "", &cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let layer = VitsResidualCouplingLayer::from_varbuilder(&client, &mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 3;
        let inputs_data: Vec<f32> = (0..cfg.flow_size * length)
            .map(|i| i as f32 * 0.1)
            .collect();
        let inputs = Tensor::<CpuRuntime>::from_slice(
            &inputs_data,
            &[batch, cfg.flow_size, length],
            &device,
        )
        .unwrap();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 0.0], &[batch, 1, length], &device)
                .unwrap();

        let out = layer.reverse(&client, &inputs, &mask, None).unwrap();
        assert_eq!(out.shape(), &[batch, cfg.flow_size, length]);
        let values: Vec<f32> = out.contiguous().unwrap().to_vec();
        // The first half passes through unchanged; only the transformed
        // second half is multiplied by the padding mask.
        let half = cfg.flow_size / 2;
        for c in 0..half {
            assert_eq!(
                values[c * length + length - 1],
                inputs_data[c * length + length - 1],
                "first half passes through: {values:?}"
            );
        }
        for c in half..cfg.flow_size {
            assert_eq!(
                values[c * length + length - 1],
                0.0,
                "second half is zeroed at padded positions: {values:?}"
            );
        }
    }

    #[test]
    fn coupling_block_reverse_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = VarMap::<CpuRuntime>::new();
        for i in 0..cfg.prior_encoder_num_flows {
            insert_layer_map(&mut map, &format!("flows.{i}."), &cfg, &device);
        }
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let block = VitsResidualCouplingBlock::from_varbuilder(&client, &mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 2;
        let inputs = Tensor::<CpuRuntime>::from_slice(
            &vec![0.1f32; cfg.flow_size * length],
            &[batch, cfg.flow_size, length],
            &device,
        )
        .unwrap();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[batch, 1, length], &device).unwrap();

        let out = block.reverse(&client, &inputs, &mask, None).unwrap();
        assert_eq!(out.shape(), &[batch, cfg.flow_size, length]);
    }

    #[test]
    fn sample_prior_latents_with_noise_matches_closed_form() {
        let (client, device) = cpu_setup();
        let means = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2, 1], &device).unwrap();
        let log_vars =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[1, 2, 1], &device).unwrap();
        let noise = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.5], &[1, 2, 1], &device).unwrap();

        let latents =
            sample_prior_latents_with_noise(&client, &means, &log_vars, 2.0, &noise).unwrap();
        // stddev = exp(0) = 1, so latents = means + noise * 2.0.
        assert_eq!(latents.to_vec::<f32>(), vec![2.0, 1.0]);
    }
}

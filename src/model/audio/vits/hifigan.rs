//! `VitsHifiGan`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsHifiGan`.
//!
//! `conv_pre` (kernel 7) feeds a stack of `upsample_rates.len()` stages,
//! each: leaky-ReLU → transposed-conv upsample → an average of
//! `resblock_kernel_sizes.len()` parallel [`HifiGanResidualBlock`]s. A final
//! leaky-ReLU (PyTorch's untouched default slope, NOT
//! `config.leaky_relu_slope` — HF calls `nn.functional.leaky_relu(hidden_states)`
//! with no slope argument here), `conv_post` (kernel 7, no bias), and
//! `tanh` produce the waveform. `conv_pre`/`conv_post` are plain
//! (never weight-norm reparameterized in HF — `apply_weight_norm`/
//! `remove_weight_norm` only ever touch `upsampler` and `resblocks`);
//! [`HifiGanResidualBlock`]'s convs and the upsampler's transposed convs go
//! through [`super::weight_norm_conv`]'s per-tensor detection.

use crate::error::{Error, Result};
use crate::model::audio::vits::hifigan_blocks::{HifiGanResidualBlock, HifiGanUpsampleLayer};
use crate::model::audio::vits::weight_norm_conv::load_weight_normed_conv_transpose1d;
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, PaddingMode, ReduceOps, ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// PyTorch's untouched `nn.functional.leaky_relu` default `negative_slope`.
/// HF's `VitsHifiGan.forward` applies exactly one leaky-ReLU with no slope
/// argument (the one right before `conv_post`); every other leaky-ReLU in
/// the decoder uses `config.leaky_relu_slope`.
const TORCH_DEFAULT_LEAKY_RELU_SLOPE: f64 = 0.01;

/// HiFi-GAN vocoder decoder: spectrogram (post-flow latents) `[batch,
/// flow_size, length]` → waveform `[batch, 1, length * prod(upsample_rates)]`.
pub struct VitsHifiGan<R: Runtime> {
    conv_pre: Conv1d<R>,
    upsampler: Vec<HifiGanUpsampleLayer<R>>,
    /// Flat-indexed: `resblocks[stage * num_kernels + k]`.
    resblocks: Vec<HifiGanResidualBlock<R>>,
    conv_post: Conv1d<R>,
    /// Present iff the checkpoint's `speaker_embedding_size > 0`.
    cond: Option<Conv1d<R>>,
    num_kernels: usize,
    leaky_relu_slope: f64,
}

impl<R: Runtime<DType = DType>> VitsHifiGan<R> {
    /// Reads `conv_pre.*`, `upsampler.{0..}.*`, `resblocks.{0..}.*`,
    /// `conv_post.*` and, iff `config.speaker_embedding_size != 0`,
    /// `cond.*`, under `vb`'s current prefix (the checkpoint's `decoder.`
    /// for the real MMS-TTS weights).
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &VitsConfig,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        if config.upsample_rates.len() != config.upsample_kernel_sizes.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "VitsHifiGan: upsample_rates ({}) and upsample_kernel_sizes ({}) must have \
                     the same length",
                    config.upsample_rates.len(),
                    config.upsample_kernel_sizes.len()
                ),
            });
        }
        if config.resblock_kernel_sizes.len() != config.resblock_dilation_sizes.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "VitsHifiGan: resblock_kernel_sizes ({}) and resblock_dilation_sizes ({}) \
                     must have the same length",
                    config.resblock_kernel_sizes.len(),
                    config.resblock_dilation_sizes.len()
                ),
            });
        }
        let num_kernels = config.resblock_kernel_sizes.len();

        let mut pre_vb = vb.pp("conv_pre");
        let pre_weight = pre_vb.take_tensor("weight")?;
        let pre_bias = pre_vb.take_tensor("bias")?;
        drop(pre_vb);
        let conv_pre = Conv1d::new(
            pre_weight,
            Some(pre_bias),
            1,
            PaddingMode::conv1d(3, 3),
            1,
            1,
            false,
        );

        let mut upsampler = Vec::with_capacity(config.upsample_rates.len());
        for (i, (&rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let mut up_vb = vb.pp("upsampler");
            let (weight, bias) =
                load_weight_normed_conv_transpose1d(client, &mut up_vb, &i.to_string())?;
            drop(up_vb);
            let padding = (kernel_size - rate) / 2;
            upsampler.push(HifiGanUpsampleLayer {
                weight,
                bias,
                stride: rate,
                padding: PaddingMode::conv1d(padding, padding),
            });
        }

        let mut resblocks = Vec::with_capacity(upsampler.len() * num_kernels);
        // Channel width per stage (`upsample_initial_channel / 2^(stage+1)`
        // in HF) is implied by each loaded conv's own weight shape, so it
        // is never separately computed or asserted here.
        for _stage in 0..upsampler.len() {
            for (kernel_size, dilations) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
            {
                let idx = resblocks.len();
                let mut block_vb = vb.pp(&format!("resblocks.{idx}"));
                resblocks.push(HifiGanResidualBlock::from_varbuilder(
                    client,
                    &mut block_vb,
                    *kernel_size,
                    dilations,
                    config.leaky_relu_slope,
                )?);
                drop(block_vb);
            }
        }

        let mut post_vb = vb.pp("conv_post");
        let post_weight = post_vb.take_tensor("weight")?;
        drop(post_vb);
        let conv_post = Conv1d::new(post_weight, None, 1, PaddingMode::conv1d(3, 3), 1, 1, false);

        let cond = if config.speaker_embedding_size != 0 {
            let mut cond_vb = vb.pp("cond");
            let weight = cond_vb.take_tensor("weight")?;
            let bias = cond_vb.take_tensor("bias")?;
            drop(cond_vb);
            Some(Conv1d::new(
                weight,
                Some(bias),
                1,
                PaddingMode::Valid,
                1,
                1,
                false,
            ))
        } else {
            None
        };

        Ok(Self {
            conv_pre,
            upsampler,
            resblocks,
            conv_post,
            cond,
            num_kernels,
            leaky_relu_slope: config.leaky_relu_slope,
        })
    }

    /// `spectrogram`: `[batch, flow_size, length]` (the masked flow output).
    /// `global_conditioning`: `[batch, speaker_embedding_size, 1]`, required
    /// iff this instance has a `cond` layer. Returns the waveform `[batch,
    /// 1, length * prod(upsample_rates)]`.
    pub fn forward<C>(
        &self,
        client: &C,
        spectrogram: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + ConvOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>,
    {
        let mut hidden_states = self.conv_pre.forward_inference(client, spectrogram)?;

        hidden_states = match (global_conditioning, &self.cond) {
            (Some(g), Some(cond)) => {
                let g = cond.forward_inference(client, g)?;
                client.add(&hidden_states, &g).map_err(Error::Numr)?
            }
            (Some(_), None) => {
                return Err(Error::ModelError {
                    reason: "VitsHifiGan: global_conditioning given but no cond layer loaded \
                             (speaker_embedding_size == 0)"
                        .into(),
                });
            }
            (None, _) => hidden_states,
        };

        for (stage, upsample) in self.upsampler.iter().enumerate() {
            hidden_states = client
                .leaky_relu(&hidden_states, self.leaky_relu_slope)
                .map_err(Error::Numr)?;
            hidden_states = upsample.forward(client, &hidden_states)?;

            let base = stage * self.num_kernels;
            let mut res_state = self.resblocks[base].forward(client, &hidden_states)?;
            for k in 1..self.num_kernels {
                let out = self.resblocks[base + k].forward(client, &hidden_states)?;
                res_state = client.add(&res_state, &out).map_err(Error::Numr)?;
            }
            hidden_states = client
                .mul_scalar(&res_state, 1.0 / self.num_kernels as f64)
                .map_err(Error::Numr)?;
        }

        let hidden_states = client
            .leaky_relu(&hidden_states, TORCH_DEFAULT_LEAKY_RELU_SLOPE)
            .map_err(Error::Numr)?;
        let hidden_states = self.conv_post.forward_inference(client, &hidden_states)?;
        client.tanh(&hidden_states).map_err(Error::Numr)
    }
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
            upsample_initial_channel: 8,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            leaky_relu_slope: 0.1,
            speaker_embedding_size: 0,
            ..Default::default()
        }
    }

    fn build_map(cfg: &VitsConfig, device: &numr::runtime::cpu::CpuDevice) -> VarMap<CpuRuntime> {
        let mut map = VarMap::<CpuRuntime>::new();
        let flow = cfg.flow_size;
        let c0 = cfg.upsample_initial_channel;
        map.insert(
            "conv_pre.weight".into(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.05f32; c0 * flow * 7], &[c0, flow, 7], device)
                .unwrap(),
        );
        map.insert(
            "conv_pre.bias".into(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c0], &[c0], device).unwrap(),
        );

        let mut channels = c0;
        let mut resblock_idx = 0;
        for (i, (&rate, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_channels = channels / 2;
            map.insert(
                format!("upsampler.{i}.weight"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.05f32; channels * out_channels * k],
                    &[channels, out_channels, k],
                    device,
                )
                .unwrap(),
            );
            map.insert(
                format!("upsampler.{i}.bias"),
                Tensor::<CpuRuntime>::from_slice(
                    &vec![0.0f32; out_channels],
                    &[out_channels],
                    device,
                )
                .unwrap(),
            );
            let _ = rate;

            for (kernel_size, dilations) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
            {
                for (j, &_dilation) in dilations.iter().enumerate() {
                    map.insert(
                        format!("resblocks.{resblock_idx}.convs1.{j}.weight"),
                        Tensor::<CpuRuntime>::from_slice(
                            &vec![0.05f32; out_channels * out_channels * kernel_size],
                            &[out_channels, out_channels, *kernel_size],
                            device,
                        )
                        .unwrap(),
                    );
                    map.insert(
                        format!("resblocks.{resblock_idx}.convs1.{j}.bias"),
                        Tensor::<CpuRuntime>::from_slice(
                            &vec![0.0f32; out_channels],
                            &[out_channels],
                            device,
                        )
                        .unwrap(),
                    );
                    map.insert(
                        format!("resblocks.{resblock_idx}.convs2.{j}.weight"),
                        Tensor::<CpuRuntime>::from_slice(
                            &vec![0.05f32; out_channels * out_channels * kernel_size],
                            &[out_channels, out_channels, *kernel_size],
                            device,
                        )
                        .unwrap(),
                    );
                    map.insert(
                        format!("resblocks.{resblock_idx}.convs2.{j}.bias"),
                        Tensor::<CpuRuntime>::from_slice(
                            &vec![0.0f32; out_channels],
                            &[out_channels],
                            device,
                        )
                        .unwrap(),
                    );
                }
                resblock_idx += 1;
            }
            channels = out_channels;
        }

        map.insert(
            "conv_post.weight".into(),
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.05f32; channels * 7],
                &[1, channels, 7],
                device,
            )
            .unwrap(),
        );
        map
    }

    #[test]
    fn forward_upsamples_by_product_of_rates_and_masks_shape() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = build_map(&cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let decoder = VitsHifiGan::from_varbuilder(&client, &mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 3;
        let spectrogram = Tensor::<CpuRuntime>::from_slice(
            &vec![0.1f32; cfg.flow_size * length],
            &[batch, cfg.flow_size, length],
            &device,
        )
        .unwrap();

        let waveform = decoder.forward(&client, &spectrogram, None).unwrap();
        let expected_rate: usize = cfg.upsample_rates.iter().product();
        assert_eq!(waveform.shape(), &[batch, 1, length * expected_rate]);
        let values: Vec<f32> = waveform.contiguous().unwrap().to_vec();
        assert!(values.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn forward_rejects_conditioning_without_cond_layer() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = build_map(&cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let decoder = VitsHifiGan::from_varbuilder(&client, &mut vb, &cfg).unwrap();

        let spectrogram = Tensor::<CpuRuntime>::from_slice(
            &vec![0.1f32; cfg.flow_size],
            &[1, cfg.flow_size, 1],
            &device,
        )
        .unwrap();
        let bogus_cond = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();

        let err = decoder.forward(&client, &spectrogram, Some(&bogus_cond));
        assert!(err.is_err());
    }
}

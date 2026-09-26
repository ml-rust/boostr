//! `VitsStochasticDurationPredictor`, reverse (inference) path only.
//!
//! Exact port of HF
//! `transformers.models.vits.modeling_vits.VitsStochasticDurationPredictor`.
//! The training-time forward (posterior flow, `post_conv_*`/`post_flows`,
//! the NLL/logq loss) is not ported: this crate is inference-only, and that
//! branch is unreachable from a loaded checkpoint at inference time.

use crate::error::{Error, Result};
use crate::model::audio::vits::dds_conv::VitsDilatedDepthSeparableConv;
use crate::model::audio::vits::flows::{VitsConvFlow, VitsElementwiseAffine};
use crate::model::audio::vits::spline::RqSplineClient;
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ConvOps, NormalizationOps, PaddingMode, RandomOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One step of the reverse flow stack, in the order the reverse path
/// actually runs them (see [`VitsStochasticDurationPredictor::from_varbuilder`]).
enum DurationFlow<R: Runtime> {
    Affine(Box<VitsElementwiseAffine<R>>),
    Conv(Box<VitsConvFlow<R>>),
}

/// Flow-based (stochastic) duration predictor. Reverse path draws a
/// `[batch, 2, length]` Gaussian, runs it backward through the flow stack,
/// and returns the first channel as `log_duration`.
pub struct VitsStochasticDurationPredictor<R: Runtime> {
    conv_pre: Conv1d<R>,
    conv_proj: Conv1d<R>,
    conv_dds: VitsDilatedDepthSeparableConv<R>,
    /// Present iff the checkpoint's `speaker_embedding_size > 0`.
    cond: Option<Conv1d<R>>,
    /// HF builds `self.flows = [ElementwiseAffine, ConvFlow x N]`, then the
    /// reverse path runs `reversed(flows)[:-2] + [reversed(flows)[-1]]` —
    /// net effect: drop `ConvFlow_1`, keep the rest in reverse order, run
    /// `ElementwiseAffine` last. Precomputed once at load time as
    /// `[ConvFlow_N, ..., ConvFlow_2, ElementwiseAffine]`.
    reverse_flows: Vec<DurationFlow<R>>,
}

impl<R: Runtime<DType = DType>> VitsStochasticDurationPredictor<R> {
    /// Reads `conv_pre.*`, `conv_proj.*`, `conv_dds.*`, `flows.{0..=num_flows}.*`
    /// and, iff `config.speaker_embedding_size != 0`, `cond.*`, under `vb`'s
    /// current prefix (the checkpoint's `duration_predictor.` for the real
    /// MMS-TTS weights). Does NOT read `post_conv_*`/`post_flows` — those
    /// back the training-only forward branch this crate never runs.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let filter_channels = config.hidden_size;

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

        let mut proj_vb = vb.pp("conv_proj");
        let proj_weight = proj_vb.take_tensor("weight")?;
        let proj_bias = proj_vb.take_tensor("bias")?;
        drop(proj_vb);
        let conv_proj = Conv1d::new(
            proj_weight,
            Some(proj_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        let mut dds_vb = vb.pp("conv_dds");
        let conv_dds = VitsDilatedDepthSeparableConv::from_varbuilder(
            &mut dds_vb,
            filter_channels,
            config.duration_predictor_kernel_size,
            config.depth_separable_num_layers,
        )?;
        drop(dds_vb);

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

        let mut affine_vb = vb.pp("flows.0");
        let elementwise_affine = VitsElementwiseAffine::from_varbuilder(&mut affine_vb)?;
        drop(affine_vb);

        let num_flows = config.duration_predictor_num_flows;
        let mut conv_flows = Vec::with_capacity(num_flows);
        for i in 1..=num_flows {
            let mut flow_vb = vb.pp(&format!("flows.{i}"));
            conv_flows.push(VitsConvFlow::from_varbuilder(&mut flow_vb, config)?);
            drop(flow_vb);
        }

        // Drop ConvFlow_1 (index 0 here), reverse the rest, then run
        // ElementwiseAffine last — see the `reverse_flows` field doc.
        let mut reverse_flows = Vec::with_capacity(num_flows);
        for conv_flow in conv_flows.into_iter().skip(1).rev() {
            reverse_flows.push(DurationFlow::Conv(Box::new(conv_flow)));
        }
        reverse_flows.push(DurationFlow::Affine(Box::new(elementwise_affine)));

        Ok(Self {
            conv_pre,
            conv_proj,
            conv_dds,
            cond,
            reverse_flows,
        })
    }

    /// Draws `noise` via [`RandomOps::randn_seeded`] and runs
    /// [`Self::forward_reverse_with_noise`]. The normal inference entry
    /// point; `hidden_states`: `[batch, hidden_size, length]`, `seed` picks
    /// the deterministic PRNG stream (same seed + shape + backend always
    /// reproduces the same draw, see [`RandomOps::randn_seeded`]).
    pub fn forward_reverse<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
        noise_scale: f64,
        seed: u64,
    ) -> Result<Tensor<R>>
    where
        C: RqSplineClient<R> + ConvOps<R> + NormalizationOps<R> + RandomOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = [hidden_states.shape()[0], 2, hidden_states.shape()[2]];
        let noise = client
            .randn_seeded(&shape, hidden_states.dtype(), seed)
            .map_err(Error::Numr)?;
        self.forward_reverse_with_noise(
            client,
            hidden_states,
            padding_mask,
            global_conditioning,
            noise_scale,
            &noise,
        )
    }

    /// Reverse path with a caller-supplied noise draw in place of a fresh
    /// [`RandomOps::randn_seeded`] draw. `noise`: `[batch, 2, length]`,
    /// standard normal. Exists so a parity test can replay a captured
    /// `torch.randn(batch, 2, length)` draw bit-for-bit instead of a
    /// reproduced-but-necessarily-different RNG stream;
    /// [`Self::forward_reverse`] is the normal entry point.
    ///
    /// Returns `log_duration`: `[batch, 1, length]`.
    pub fn forward_reverse_with_noise<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
        noise_scale: f64,
        noise: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RqSplineClient<R> + ConvOps<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let inputs = self.conv_pre.forward_inference(client, hidden_states)?;
        let inputs = match (global_conditioning, &self.cond) {
            (Some(g), Some(cond)) => {
                let g = cond.forward_inference(client, g)?;
                client.add(&inputs, &g).map_err(Error::Numr)?
            }
            (Some(_), None) => {
                return Err(Error::ModelError {
                    reason: "VitsStochasticDurationPredictor: global_conditioning given but no \
                             cond layer loaded (speaker_embedding_size == 0)"
                        .into(),
                });
            }
            (None, _) => inputs,
        };
        let inputs = self.conv_dds.forward(client, &inputs, padding_mask, None)?;
        let inputs = client
            .mul(
                &self.conv_proj.forward_inference(client, &inputs)?,
                padding_mask,
            )
            .map_err(Error::Numr)?;

        let mut latents = client.mul_scalar(noise, noise_scale).map_err(Error::Numr)?;
        for flow in &self.reverse_flows {
            latents = latents
                .flip(1)
                .and_then(|t| t.contiguous())
                .map_err(Error::Numr)?;
            latents = match flow {
                DurationFlow::Affine(affine) => affine.reverse(client, &latents, padding_mask)?,
                DurationFlow::Conv(conv) => {
                    conv.reverse(client, &latents, padding_mask, &inputs)?
                }
            };
        }

        latents
            .narrow(1, 0, 1)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tiny_config() -> VitsConfig {
        VitsConfig {
            hidden_size: 4,
            depth_separable_channels: 2,
            depth_separable_num_layers: 1,
            duration_predictor_kernel_size: 3,
            duration_predictor_flow_bins: 3,
            duration_predictor_tail_bound: 5.0,
            duration_predictor_num_flows: 2,
            layer_norm_eps: 1e-5,
            speaker_embedding_size: 0,
            ..Default::default()
        }
    }

    fn put(
        map: &mut VarMap<CpuRuntime>,
        name: &str,
        data: Vec<f32>,
        shape: &[usize],
        device: &CpuDevice,
    ) {
        map.insert(
            name.to_string(),
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap(),
        );
    }

    fn build_map(cfg: &VitsConfig, device: &CpuDevice) -> VarMap<CpuRuntime> {
        let h = cfg.hidden_size;
        let mut map = VarMap::<CpuRuntime>::new();
        put(
            &mut map,
            "conv_pre.weight",
            vec![0.05; h * h],
            &[h, h, 1],
            device,
        );
        put(&mut map, "conv_pre.bias", vec![0.0; h], &[h], device);
        put(
            &mut map,
            "conv_proj.weight",
            vec![0.05; h * h],
            &[h, h, 1],
            device,
        );
        put(&mut map, "conv_proj.bias", vec![0.0; h], &[h], device);

        let k = cfg.duration_predictor_kernel_size;
        for i in 0..cfg.depth_separable_num_layers {
            let p = format!("conv_dds.convs_dilated.{i}");
            put(
                &mut map,
                &format!("{p}.weight"),
                vec![0.1; h * k],
                &[h, 1, k],
                device,
            );
            put(&mut map, &format!("{p}.bias"), vec![0.0; h], &[h], device);
            let p = format!("conv_dds.convs_pointwise.{i}");
            put(
                &mut map,
                &format!("{p}.weight"),
                vec![0.1; h * h],
                &[h, h, 1],
                device,
            );
            put(&mut map, &format!("{p}.bias"), vec![0.0; h], &[h], device);
            put(
                &mut map,
                &format!("conv_dds.norms_1.{i}.weight"),
                vec![1.0; h],
                &[h],
                device,
            );
            put(
                &mut map,
                &format!("conv_dds.norms_1.{i}.bias"),
                vec![0.0; h],
                &[h],
                device,
            );
            put(
                &mut map,
                &format!("conv_dds.norms_2.{i}.weight"),
                vec![1.0; h],
                &[h],
                device,
            );
            put(
                &mut map,
                &format!("conv_dds.norms_2.{i}.bias"),
                vec![0.0; h],
                &[h],
                device,
            );
        }

        put(&mut map, "flows.0.translate", vec![0.0; 2], &[2, 1], device);
        put(&mut map, "flows.0.log_scale", vec![0.0; 2], &[2, 1], device);

        let num_bins = cfg.duration_predictor_flow_bins;
        let bin_params = 3 * num_bins - 1;
        for i in 1..=cfg.duration_predictor_num_flows {
            let p = format!("flows.{i}");
            put(
                &mut map,
                &format!("{p}.conv_pre.weight"),
                vec![0.05; h],
                &[h, 1, 1],
                device,
            );
            put(
                &mut map,
                &format!("{p}.conv_pre.bias"),
                vec![0.0; h],
                &[h],
                device,
            );
            for j in 0..cfg.depth_separable_num_layers {
                let dp = format!("{p}.conv_dds.convs_dilated.{j}");
                put(
                    &mut map,
                    &format!("{dp}.weight"),
                    vec![0.1; h * k],
                    &[h, 1, k],
                    device,
                );
                put(&mut map, &format!("{dp}.bias"), vec![0.0; h], &[h], device);
                let dp = format!("{p}.conv_dds.convs_pointwise.{j}");
                put(
                    &mut map,
                    &format!("{dp}.weight"),
                    vec![0.1; h * h],
                    &[h, h, 1],
                    device,
                );
                put(&mut map, &format!("{dp}.bias"), vec![0.0; h], &[h], device);
                put(
                    &mut map,
                    &format!("{p}.conv_dds.norms_1.{j}.weight"),
                    vec![1.0; h],
                    &[h],
                    device,
                );
                put(
                    &mut map,
                    &format!("{p}.conv_dds.norms_1.{j}.bias"),
                    vec![0.0; h],
                    &[h],
                    device,
                );
                put(
                    &mut map,
                    &format!("{p}.conv_dds.norms_2.{j}.weight"),
                    vec![1.0; h],
                    &[h],
                    device,
                );
                put(
                    &mut map,
                    &format!("{p}.conv_dds.norms_2.{j}.bias"),
                    vec![0.0; h],
                    &[h],
                    device,
                );
            }
            put(
                &mut map,
                &format!("{p}.conv_proj.weight"),
                vec![0.01; bin_params * h],
                &[bin_params, h, 1],
                device,
            );
            put(
                &mut map,
                &format!("{p}.conv_proj.bias"),
                vec![0.0; bin_params],
                &[bin_params],
                device,
            );
        }
        map
    }

    #[test]
    fn reverse_produces_expected_shape_and_masks_padding() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = build_map(&cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let predictor = VitsStochasticDurationPredictor::from_varbuilder(&mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 3;
        let hidden_data: Vec<f32> = (0..cfg.hidden_size * length)
            .map(|i| i as f32 * 0.1)
            .collect();
        let hidden_states = Tensor::<CpuRuntime>::from_slice(
            &hidden_data,
            &[batch, cfg.hidden_size, length],
            &device,
        )
        .unwrap();
        // last position is padding.
        let padding_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 0.0], &[batch, 1, length], &device)
                .unwrap();
        let noise = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32, -0.2, 0.3, 0.4, -0.1, 0.05],
            &[batch, 2, length],
            &device,
        )
        .unwrap();

        let log_duration = predictor
            .forward_reverse_with_noise(&client, &hidden_states, &padding_mask, None, 0.8, &noise)
            .unwrap();
        assert_eq!(log_duration.shape(), &[batch, 1, length]);
        let values: Vec<f32> = log_duration.to_vec();
        assert_eq!(values[2], 0.0, "padded position must be zeroed: {values:?}");
    }

    #[test]
    fn reverse_rejects_conditioning_without_cond_layer() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = build_map(&cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let predictor = VitsStochasticDurationPredictor::from_varbuilder(&mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 2;
        let hidden_states = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; cfg.hidden_size * length],
            &[batch, cfg.hidden_size, length],
            &device,
        )
        .unwrap();
        let padding_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[batch, 1, length], &device).unwrap();
        let noise =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[batch, 2, length], &device).unwrap();
        let bogus_cond = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();

        let err = predictor.forward_reverse_with_noise(
            &client,
            &hidden_states,
            &padding_mask,
            Some(&bogus_cond),
            0.8,
            &noise,
        );
        assert!(err.is_err());
    }
}

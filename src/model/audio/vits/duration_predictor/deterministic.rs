//! `VitsDurationPredictor`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsDurationPredictor`.
//!
//! Two masked conv + `LayerNorm` + `ReLU` blocks, then a `1x1` projection to
//! a single `log_duration` channel. Loaded only when
//! [`VitsConfig::use_stochastic_duration_prediction`] is `false` — no
//! published MMS-TTS checkpoint sets this, but the config field selects it.
//! Dropout is a training-only no-op here (inference-only module).

use crate::error::{Error, Result};
use crate::model::audio::vits::layer_norm_channels::layer_norm_channels;
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, LayerNorm, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, PaddingMode, ScalarOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsDurationPredictor<R: Runtime> {
    conv_1: Conv1d<R>,
    norm_1: LayerNorm<R>,
    conv_2: Conv1d<R>,
    norm_2: LayerNorm<R>,
    proj: Conv1d<R>,
    /// Present iff the checkpoint's `speaker_embedding_size > 0`.
    cond: Option<Conv1d<R>>,
}

impl<R: Runtime<DType = DType>> VitsDurationPredictor<R> {
    /// Reads `conv_1.*`, `norm_1.*`, `conv_2.*`, `norm_2.*`, `proj.*` and,
    /// iff `config.speaker_embedding_size != 0`, `cond.*`.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let kernel_size = config.duration_predictor_kernel_size;
        let padding = kernel_size / 2;
        let eps = config.layer_norm_eps as f32;

        let mut c1_vb = vb.pp("conv_1");
        let c1_weight = c1_vb.take_tensor("weight")?;
        let c1_bias = c1_vb.take_tensor("bias")?;
        drop(c1_vb);
        let conv_1 = Conv1d::new(
            c1_weight,
            Some(c1_bias),
            1,
            PaddingMode::Custom(padding, padding, 0, 0),
            1,
            1,
            false,
        );

        let mut n1_vb = vb.pp("norm_1");
        let n1_weight = n1_vb.take_tensor("weight")?;
        let n1_bias = n1_vb.take_tensor("bias")?;
        drop(n1_vb);
        let norm_1 = LayerNorm::new(n1_weight, n1_bias, eps, false);

        let mut c2_vb = vb.pp("conv_2");
        let c2_weight = c2_vb.take_tensor("weight")?;
        let c2_bias = c2_vb.take_tensor("bias")?;
        drop(c2_vb);
        let conv_2 = Conv1d::new(
            c2_weight,
            Some(c2_bias),
            1,
            PaddingMode::Custom(padding, padding, 0, 0),
            1,
            1,
            false,
        );

        let mut n2_vb = vb.pp("norm_2");
        let n2_weight = n2_vb.take_tensor("weight")?;
        let n2_bias = n2_vb.take_tensor("bias")?;
        drop(n2_vb);
        let norm_2 = LayerNorm::new(n2_weight, n2_bias, eps, false);

        let mut proj_vb = vb.pp("proj");
        let proj_weight = proj_vb.take_tensor("weight")?;
        let proj_bias = proj_vb.take_tensor("bias")?;
        drop(proj_vb);
        let proj = Conv1d::new(
            proj_weight,
            Some(proj_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

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
            conv_1,
            norm_1,
            conv_2,
            norm_2,
            proj,
            cond,
        })
    }

    /// `inputs`: `[batch, hidden_size, length]`. `padding_mask`: `[batch, 1,
    /// length]`. Returns `log_duration`: `[batch, 1, length]`.
    pub fn forward<C>(
        &self,
        client: &C,
        inputs: &Tensor<R>,
        padding_mask: &Tensor<R>,
        global_conditioning: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + ActivationOps<R> + NormalizationOps<R> + BinaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let inputs = match (global_conditioning, &self.cond) {
            (Some(g), Some(cond)) => {
                let g = cond.forward_inference(client, g)?;
                client.add(inputs, &g).map_err(Error::Numr)?
            }
            (Some(_), None) => {
                return Err(Error::ModelError {
                    reason: "VitsDurationPredictor: global_conditioning given but no cond layer \
                             loaded (speaker_embedding_size == 0)"
                        .into(),
                });
            }
            (None, _) => inputs.clone(),
        };

        let masked = client.mul(&inputs, padding_mask).map_err(Error::Numr)?;
        let hidden = self.conv_1.forward_inference(client, &masked)?;
        let hidden = client.relu(&hidden).map_err(Error::Numr)?;
        let hidden = layer_norm_channels(client, &self.norm_1, &hidden)?;

        let masked = client.mul(&hidden, padding_mask).map_err(Error::Numr)?;
        let hidden = self.conv_2.forward_inference(client, &masked)?;
        let hidden = client.relu(&hidden).map_err(Error::Numr)?;
        let hidden = layer_norm_channels(client, &self.norm_2, &hidden)?;

        let masked = client.mul(&hidden, padding_mask).map_err(Error::Numr)?;
        let out = self.proj.forward_inference(client, &masked)?;
        client.mul(&out, padding_mask).map_err(Error::Numr)
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
            hidden_size: 4,
            duration_predictor_kernel_size: 3,
            duration_predictor_filter_channels: 4,
            layer_norm_eps: 1e-5,
            speaker_embedding_size: 0,
            ..Default::default()
        }
    }

    #[test]
    fn forward_produces_expected_shape_and_masks_padding() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let f = cfg.duration_predictor_filter_channels;
        let k = cfg.duration_predictor_kernel_size;

        let mut map = VarMap::<CpuRuntime>::new();
        let mut put = |name: &str, data: Vec<f32>, shape: &[usize]| {
            map.insert(
                name.to_string(),
                Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap(),
            );
        };
        put("conv_1.weight", vec![0.1; f * h * k], &[f, h, k]);
        put("conv_1.bias", vec![0.0; f], &[f]);
        put("norm_1.weight", vec![1.0; f], &[f]);
        put("norm_1.bias", vec![0.0; f], &[f]);
        put("conv_2.weight", vec![0.1; f * f * k], &[f, f, k]);
        put("conv_2.bias", vec![0.0; f], &[f]);
        put("norm_2.weight", vec![1.0; f], &[f]);
        put("norm_2.bias", vec![0.0; f], &[f]);
        put("proj.weight", vec![0.1; f], &[1, f, 1]);
        put("proj.bias", vec![0.0], &[1]);

        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let predictor = VitsDurationPredictor::from_varbuilder(&mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 3;
        let inputs_data: Vec<f32> = (0..h * length).map(|i| i as f32 * 0.1).collect();
        let inputs =
            Tensor::<CpuRuntime>::from_slice(&inputs_data, &[batch, h, length], &device).unwrap();
        let padding_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 0.0], &[batch, 1, length], &device)
                .unwrap();

        let log_duration = predictor
            .forward(&client, &inputs, &padding_mask, None)
            .unwrap();
        assert_eq!(log_duration.shape(), &[batch, 1, length]);
        let values: Vec<f32> = log_duration.to_vec();
        assert_eq!(values[2], 0.0, "padded position must be zeroed: {values:?}");
    }

    #[test]
    fn forward_rejects_conditioning_without_cond_layer() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let f = cfg.duration_predictor_filter_channels;
        let k = cfg.duration_predictor_kernel_size;

        let mut map = VarMap::<CpuRuntime>::new();
        let mut put = |name: &str, data: Vec<f32>, shape: &[usize]| {
            map.insert(
                name.to_string(),
                Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap(),
            );
        };
        put("conv_1.weight", vec![0.1; f * h * k], &[f, h, k]);
        put("conv_1.bias", vec![0.0; f], &[f]);
        put("norm_1.weight", vec![1.0; f], &[f]);
        put("norm_1.bias", vec![0.0; f], &[f]);
        put("conv_2.weight", vec![0.1; f * f * k], &[f, f, k]);
        put("conv_2.bias", vec![0.0; f], &[f]);
        put("norm_2.weight", vec![1.0; f], &[f]);
        put("norm_2.bias", vec![0.0; f], &[f]);
        put("proj.weight", vec![0.1; f], &[1, f, 1]);
        put("proj.bias", vec![0.0], &[1]);

        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let predictor = VitsDurationPredictor::from_varbuilder(&mut vb, &cfg).unwrap();

        let batch = 1;
        let length = 2;
        let inputs = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; h * length],
            &[batch, h, length],
            &device,
        )
        .unwrap();
        let padding_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 2], &[batch, 1, length], &device).unwrap();
        let bogus_cond = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();

        let err = predictor.forward(&client, &inputs, &padding_mask, Some(&bogus_cond));
        assert!(err.is_err());
    }
}

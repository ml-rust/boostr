//! `VitsTextEncoder`: exact port of HF
//! `transformers.models.vits.modeling_vits.VitsTextEncoder`.
//!
//! `embed_tokens(input_ids) * sqrt(hidden_size)` (HF's `VitsTextEncoder`
//! scales the raw embedding lookup before the transformer stack — unlike a
//! `padding_idx` argument, this scale changes forward values, so it must be
//! reproduced, not dropped as a training-only detail), through
//! [`VitsEncoder`], then a `1x1` conv (`project`) split down the channel axis
//! into `prior_means` and `prior_log_variances`.

use crate::error::{Error, Result};
use crate::model::audio::vits::encoder::VitsEncoder;
use crate::model::config::VitsConfig;
use crate::nn::{Conv1d, Embedding, VarBuilder};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps, PaddingMode,
    ScalarOps, ShapeOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `last_hidden_state`: `[batch, seq, hidden]`. `prior_means` /
/// `prior_log_variances`: `[batch, seq, flow_size]` each.
pub struct VitsTextEncoderOutput<R: Runtime> {
    pub last_hidden_state: Tensor<R>,
    pub prior_means: Tensor<R>,
    pub prior_log_variances: Tensor<R>,
}

pub struct VitsTextEncoder<R: Runtime> {
    embed_tokens: Embedding<R>,
    encoder: VitsEncoder<R>,
    project: Conv1d<R>,
    hidden_size: usize,
    flow_size: usize,
}

impl<R: Runtime<DType = DType>> VitsTextEncoder<R> {
    /// Reads `embed_tokens.weight`, `encoder.*`, `project.{weight,bias}`
    /// under `vb`'s current prefix (the checkpoint's `text_encoder.` for the
    /// real MMS-TTS weights).
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &VitsConfig) -> Result<Self> {
        let mut emb_vb = vb.pp("embed_tokens");
        let emb_weight = emb_vb.take_tensor("weight")?;
        drop(emb_vb);
        let embed_tokens = Embedding::new(emb_weight, false);

        let mut enc_vb = vb.pp("encoder");
        let encoder = VitsEncoder::from_varbuilder(&mut enc_vb, config)?;
        drop(enc_vb);

        let mut proj_vb = vb.pp("project");
        let proj_weight = proj_vb.take_tensor("weight")?;
        let proj_bias = proj_vb.take_tensor("bias")?;
        drop(proj_vb);
        let project = Conv1d::new(
            proj_weight,
            Some(proj_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        Ok(Self {
            embed_tokens,
            encoder,
            project,
            hidden_size: config.hidden_size,
            flow_size: config.flow_size,
        })
    }

    /// `input_ids`: `[batch, seq]`, integer dtype. `padding_mask`: `[batch,
    /// seq, 1]` (1.0 valid, 0.0 padded) — HF's `input_padding_mask`.
    /// `attention_mask`, if given: `[batch, seq]`, same convention, passed
    /// through to [`VitsEncoder`] for the additive self-attention mask.
    pub fn forward<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        padding_mask: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<VitsTextEncoderOutput<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let embedded = self.embed_tokens.forward(client, input_ids)?;
        let scale = (self.hidden_size as f64).sqrt();
        let hidden_states = client
            .mul_scalar(embedded.tensor(), scale)
            .map_err(Error::Numr)?;

        let last_hidden_state =
            self.encoder
                .forward(client, &hidden_states, padding_mask, attention_mask)?;

        let transposed = last_hidden_state
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let stats = self.project.forward_inference(client, &transposed)?;
        let stats = stats
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let stats = client.mul(&stats, padding_mask).map_err(Error::Numr)?;

        let prior_means = stats
            .narrow(2, 0, self.flow_size)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let prior_log_variances = stats
            .narrow(2, self.flow_size, self.flow_size)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        Ok(VitsTextEncoderOutput {
            last_hidden_state,
            prior_means,
            prior_log_variances,
        })
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
            vocab_size: 6,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            window_size: 0,
            use_bias: true,
            ffn_dim: 8,
            ffn_kernel_size: 3,
            flow_size: 4,
            hidden_act: "relu".to_string(),
            layer_norm_eps: 1e-5,
            ..Default::default()
        }
    }

    fn build_map(cfg: &VitsConfig, device: &numr::runtime::cpu::CpuDevice) -> VarMap<CpuRuntime> {
        let h = cfg.hidden_size;
        let ffn = cfg.ffn_dim;
        let k = cfg.ffn_kernel_size;
        let flow2 = cfg.flow_size * 2;
        let mut map = VarMap::<CpuRuntime>::new();
        let mut put = |name: &str, data: Vec<f32>, shape: &[usize]| {
            map.insert(
                name.to_string(),
                Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap(),
            );
        };
        put(
            "embed_tokens.weight",
            (0..cfg.vocab_size * h).map(|i| i as f32 * 0.01).collect(),
            &[cfg.vocab_size, h],
        );
        for i in 0..cfg.num_hidden_layers {
            let p = format!("encoder.layers.{i}");
            put(
                &format!("{p}.attention.q_proj.weight"),
                vec![0.1; h * h],
                &[h, h],
            );
            put(&format!("{p}.attention.q_proj.bias"), vec![0.0; h], &[h]);
            put(
                &format!("{p}.attention.k_proj.weight"),
                vec![0.1; h * h],
                &[h, h],
            );
            put(&format!("{p}.attention.k_proj.bias"), vec![0.0; h], &[h]);
            put(
                &format!("{p}.attention.v_proj.weight"),
                vec![0.1; h * h],
                &[h, h],
            );
            put(&format!("{p}.attention.v_proj.bias"), vec![0.0; h], &[h]);
            put(
                &format!("{p}.attention.out_proj.weight"),
                vec![0.1; h * h],
                &[h, h],
            );
            put(&format!("{p}.attention.out_proj.bias"), vec![0.0; h], &[h]);
            put(&format!("{p}.layer_norm.weight"), vec![1.0; h], &[h]);
            put(&format!("{p}.layer_norm.bias"), vec![0.0; h], &[h]);
            put(
                &format!("{p}.feed_forward.conv_1.weight"),
                vec![0.1; ffn * h * k],
                &[ffn, h, k],
            );
            put(
                &format!("{p}.feed_forward.conv_1.bias"),
                vec![0.0; ffn],
                &[ffn],
            );
            put(
                &format!("{p}.feed_forward.conv_2.weight"),
                vec![0.1; h * ffn * k],
                &[h, ffn, k],
            );
            put(&format!("{p}.feed_forward.conv_2.bias"), vec![0.0; h], &[h]);
            put(&format!("{p}.final_layer_norm.weight"), vec![1.0; h], &[h]);
            put(&format!("{p}.final_layer_norm.bias"), vec![0.0; h], &[h]);
        }
        put("project.weight", vec![0.1; flow2 * h], &[flow2, h, 1]);
        put("project.bias", vec![0.0; flow2], &[flow2]);
        map
    }

    #[test]
    fn forward_produces_expected_shapes_and_masks_padding() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let mut map = build_map(&cfg, &device);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let text_encoder = VitsTextEncoder::from_varbuilder(&mut vb, &cfg).unwrap();

        let bsz = 2;
        let seq = 3;
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 0], &[bsz, seq], &device).unwrap();
        // batch 1's last position is padding.
        let padding_mask = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0],
            &[bsz, seq, 1],
            &device,
        )
        .unwrap();

        let out = text_encoder
            .forward(&client, &input_ids, &padding_mask, None)
            .unwrap();
        assert_eq!(out.last_hidden_state.shape(), &[bsz, seq, cfg.hidden_size]);
        assert_eq!(out.prior_means.shape(), &[bsz, seq, cfg.flow_size]);
        assert_eq!(out.prior_log_variances.shape(), &[bsz, seq, cfg.flow_size]);

        let means: Vec<f32> = out.prior_means.to_vec();
        let last_row = &means[(bsz - 1) * seq * cfg.flow_size + (seq - 1) * cfg.flow_size
            ..(bsz - 1) * seq * cfg.flow_size + seq * cfg.flow_size];
        assert!(
            last_row.iter().all(|&v| v == 0.0),
            "padded position must be zeroed: {last_row:?}"
        );
    }
}

//! Stable parameter identity support for MLA.

use super::{Mla, MlaConfig};
use crate::error::{Error, Result};
use crate::nn::module::Module;
use crate::nn::{MaybeQuantLinear, RmsNorm, RoPE};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::TensorId;

/// Child modules required to construct an MLA layer while preserving parameter IDs.
///
/// Build standard children with `Linear::with_ids` / `RmsNorm::with_id` before
/// passing them here. Quantized `MaybeQuantLinear` children expose no trainable
/// `Var` parameters and are treated as inference-only.
pub struct MlaWeights<R: Runtime> {
    pub q_down: Option<MaybeQuantLinear<R>>,
    pub q_up: MaybeQuantLinear<R>,
    pub q_norm: Option<RmsNorm<R>>,
    pub kv_compress: MaybeQuantLinear<R>,
    pub kv_norm: Option<RmsNorm<R>>,
    pub kv_decompress: MaybeQuantLinear<R>,
    pub o_proj: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> Mla<R> {
    /// Create MLA from child modules while preserving IDs embedded in those children.
    pub fn with_ids(
        config: &MlaConfig,
        weights: MlaWeights<R>,
        device: &R::Device,
    ) -> Result<Self> {
        config.validate()?;
        validate_mla_weights(config, &weights)?;

        let qk_dim = config.qk_head_dim();
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            config.rope_head_dim,
            config.rope_theta,
            None,
            device,
        )?;
        let scale = 1.0 / (qk_dim as f64).sqrt();

        Ok(Self {
            q_down: weights.q_down,
            q_up: weights.q_up,
            q_norm: weights.q_norm,
            kv_compress: weights.kv_compress,
            kv_norm: weights.kv_norm,
            kv_decompress: weights.kv_decompress,
            o_proj: weights.o_proj,
            rope,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            head_dim_v: config.head_dim_v,
            rope_head_dim: config.rope_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            scale,
        })
    }
}

impl<R: Runtime> Mla<R> {
    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(q_down) = &self.q_down {
            params.extend(q_down.parameters());
        }
        params.extend(self.q_up.parameters());
        if let Some(q_norm) = &self.q_norm {
            params.extend(q_norm.parameters());
        }
        params.extend(self.kv_compress.parameters());
        if let Some(kv_norm) = &self.kv_norm {
            params.extend(kv_norm.parameters());
        }
        params.extend(self.kv_decompress.parameters());
        params.extend(self.o_proj.parameters());
        params
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named parameters for uniform module traversal.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(q_down) = &self.q_down {
            extend_named(&mut params, "q_down", q_down.named_parameters());
        }
        extend_named(&mut params, "q_up", self.q_up.named_parameters());
        if let Some(q_norm) = &self.q_norm {
            extend_named(&mut params, "q_norm", q_norm.named_parameters());
        }
        extend_named(
            &mut params,
            "kv_compress",
            self.kv_compress.named_parameters(),
        );
        if let Some(kv_norm) = &self.kv_norm {
            extend_named(&mut params, "kv_norm", kv_norm.named_parameters());
        }
        extend_named(
            &mut params,
            "kv_decompress",
            self.kv_decompress.named_parameters(),
        );
        extend_named(&mut params, "o_proj", self.o_proj.named_parameters());
        params
    }
}

impl<R: Runtime> Module<R> for Mla<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Mla::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        Mla::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Mla::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        Mla::trainable_parameters(self)
    }
}

fn validate_mla_weights<R: Runtime>(config: &MlaConfig, weights: &MlaWeights<R>) -> Result<()> {
    if config.q_uses_lora() != weights.q_down.is_some() {
        return Err(Error::ModelError {
            reason: "MLA q_down presence must match q_lora_rank".into(),
        });
    }
    if !config.q_uses_lora() && weights.q_norm.is_some() {
        return Err(Error::ModelError {
            reason: "MLA q_norm requires q_lora_rank > 0".into(),
        });
    }
    if config.use_norm {
        if config.q_uses_lora() && weights.q_norm.is_none() {
            return Err(Error::ModelError {
                reason: "MLA use_norm requires q_norm when q_lora_rank > 0".into(),
            });
        }
        if weights.kv_norm.is_none() {
            return Err(Error::ModelError {
                reason: "MLA use_norm requires kv_norm".into(),
            });
        }
    } else if weights.q_norm.is_some() || weights.kv_norm.is_some() {
        return Err(Error::ModelError {
            reason: "MLA norm weights require use_norm = true".into(),
        });
    }
    Ok(())
}

fn extend_named<'a, R: Runtime>(
    params: &mut Vec<(String, &'a Var<R>)>,
    prefix: &str,
    child: Vec<(String, &'a Var<R>)>,
) {
    params.extend(
        child
            .into_iter()
            .map(|(name, var)| (format!("{prefix}.{name}"), var)),
    );
}

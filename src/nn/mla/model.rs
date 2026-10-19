//! [`Mla`]: the Multi-Head Latent Attention layer itself and its
//! construction from a fresh config or loaded weights.

use super::config::MlaConfig;
use crate::error::Result;
use crate::nn::{Linear, MaybeQuantLinear, RmsNorm, RoPE, VarBuilder};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Multi-Head Latent Attention (MLA) layer from DeepSeek-V2.
///
/// Implements low-rank KV compression with decoupled RoPE:
/// - **Q path**: optional low-rank compression (`q_down` → norm → `q_up`) or direct projection
/// - **KV path**: `kv_compress` → norm → `kv_decompress` → split into `(k_nope, v)`
/// - **RoPE**: applied separately to `q_pe` and `k_pe` (decoupled from compressed latent)
/// - **Output**: `softmax(Q·K^T / √d) · V` projected through `o_proj`
pub struct Mla<R: Runtime> {
    // Q path
    pub(super) q_down: Option<MaybeQuantLinear<R>>,
    pub(super) q_up: MaybeQuantLinear<R>,
    pub(super) q_norm: Option<RmsNorm<R>>,

    // KV path
    pub(super) kv_compress: MaybeQuantLinear<R>,
    pub(super) kv_norm: Option<RmsNorm<R>>,
    pub(super) kv_decompress: MaybeQuantLinear<R>,

    // Output
    pub(super) o_proj: MaybeQuantLinear<R>,

    // RoPE
    pub(super) rope: RoPE<R>,

    // Config
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) head_dim_v: usize,
    pub(super) rope_head_dim: usize,
    pub(super) kv_lora_rank: usize,
    pub(super) scale: f64,
}

impl<R: Runtime<DType = DType>> Mla<R> {
    /// Create MLA from config with random/zero weights (for testing/training)
    pub fn from_config(config: &MlaConfig, device: &R::Device) -> Result<Self> {
        config.validate()?;

        let h = config.hidden_size;
        let nh = config.num_heads;
        let qk_dim = config.qk_head_dim();
        let dt = DType::F32;

        let (q_down, q_up, q_norm) = if config.q_uses_lora() {
            let q_down = MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[config.q_lora_rank, h], dt, device)?,
                None,
                true,
            ));
            let q_up = MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[nh * qk_dim, config.q_lora_rank], dt, device)?,
                None,
                true,
            ));
            let q_norm = if config.use_norm {
                Some(RmsNorm::new(
                    Tensor::<R>::ones(&[config.q_lora_rank], dt, device)?,
                    config.norm_eps,
                    true,
                ))
            } else {
                None
            };
            (Some(q_down), q_up, q_norm)
        } else {
            let q_up = MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[nh * qk_dim, h], dt, device)?,
                None,
                true,
            ));
            (None, q_up, None)
        };

        let kv_compress = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<R>::zeros(&[config.kv_lora_rank + config.rope_head_dim, h], dt, device)?,
            None,
            true,
        ));
        let kv_norm = if config.use_norm {
            Some(RmsNorm::new(
                Tensor::<R>::ones(&[config.kv_lora_rank], dt, device)?,
                config.norm_eps,
                true,
            ))
        } else {
            None
        };
        let kv_decompress = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<R>::zeros(
                &[
                    nh * (config.head_dim + config.head_dim_v),
                    config.kv_lora_rank,
                ],
                dt,
                device,
            )?,
            None,
            true,
        ));

        let o_proj = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<R>::zeros(&[h, nh * config.head_dim_v], dt, device)?,
            None,
            true,
        ));

        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            config.rope_head_dim,
            config.rope_theta,
            None,
            device,
        )?;

        let scale = 1.0 / (qk_dim as f64).sqrt();

        Ok(Self {
            q_down,
            q_up,
            q_norm,
            kv_compress,
            kv_norm,
            kv_decompress,
            o_proj,
            rope,
            num_heads: nh,
            head_dim: config.head_dim,
            head_dim_v: config.head_dim_v,
            rope_head_dim: config.rope_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            scale,
        })
    }

    /// Load MLA from pretrained weights via VarBuilder
    ///
    /// Weight names follow HuggingFace DeepSeek-V2 conventions:
    /// - `q_a_proj` / `q_b_proj` (Q down/up if q_lora_rank > 0)
    /// - `q_proj` (direct Q if q_lora_rank == 0)
    /// - `q_a_layernorm`
    /// - `kv_a_proj_with_mqa` (KV compression)
    /// - `kv_a_layernorm`
    /// - `kv_b_proj` (KV decompression)
    /// - `o_proj`
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &MlaConfig) -> Result<Self> {
        config.validate()?;

        let nh = config.num_heads;
        let qk_dim = config.qk_head_dim();

        // Q path
        let (q_down, q_up, q_norm) = if config.q_uses_lora() {
            let q_down = vb.pp("q_a_proj").take_maybe_quant_linear("weight", None)?;
            let q_up = vb.pp("q_b_proj").take_maybe_quant_linear("weight", None)?;

            let q_norm = if config.use_norm {
                let mut qn_vb = vb.pp("q_a_layernorm");
                Some(RmsNorm::new(
                    qn_vb.take_tensor("weight")?,
                    config.norm_eps,
                    false,
                ))
            } else {
                None
            };
            (Some(q_down), q_up, q_norm)
        } else {
            let q_up = vb.pp("q_proj").take_maybe_quant_linear("weight", None)?;
            (None, q_up, None)
        };

        // KV path
        let kv_compress = vb
            .pp("kv_a_proj_with_mqa")
            .take_maybe_quant_linear("weight", None)?;

        let kv_norm = if config.use_norm {
            let mut kvn_vb = vb.pp("kv_a_layernorm");
            Some(RmsNorm::new(
                kvn_vb.take_tensor("weight")?,
                config.norm_eps,
                false,
            ))
        } else {
            None
        };

        let kv_decompress = vb.pp("kv_b_proj").take_maybe_quant_linear("weight", None)?;

        // Output
        let o_proj = vb.pp("o_proj").take_maybe_quant_linear("weight", None)?;

        // RoPE
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            config.rope_head_dim,
            config.rope_theta,
            None,
            vb.device(),
        )?;

        let scale = 1.0 / (qk_dim as f64).sqrt();

        Ok(Self {
            q_down,
            q_up,
            q_norm,
            kv_compress,
            kv_norm,
            kv_decompress,
            o_proj,
            rope,
            num_heads: nh,
            head_dim: config.head_dim,
            head_dim_v: config.head_dim_v,
            rope_head_dim: config.rope_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            scale,
        })
    }
}

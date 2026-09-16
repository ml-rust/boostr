//! [`Mla`]: the Multi-Head Latent Attention layer itself — construction
//! (from a fresh config or loaded weights) and its forward pass.

use super::config::MlaConfig;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use crate::nn::{Linear, MaybeQuantLinear, RmsNorm, RoPE, VarBuilder};
use crate::ops::RoPEOps;
use crate::ops::impl_generic::attention::mla::scaled_dot_product_attention_impl;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
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

    /// Forward pass: [B, S, hidden] → [B, S, hidden]
    pub fn forward<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let shape = hidden.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];
        let qk_dim = self.head_dim + self.rope_head_dim;

        // === Q path ===
        let q = if let Some(q_down) = &self.q_down {
            let q_latent = q_down.forward(client, hidden)?;
            let q_latent = if let Some(norm) = &self.q_norm {
                norm.forward(client, &q_latent)?
            } else {
                q_latent
            };
            self.q_up.forward(client, &q_latent)?
        } else {
            self.q_up.forward(client, hidden)?
        };

        // [B, S, num_heads * qk_dim] → [B, S, H, qk_dim] → [B, H, S, qk_dim]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, qk_dim]).map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;

        // Split Q into nope and pe
        let q_nope = var_narrow(&q, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let q_nope = var_contiguous(&q_nope)?;
        let q_pe = var_narrow(&q, 3, self.head_dim, self.rope_head_dim).map_err(Error::Numr)?;
        let q_pe = var_contiguous(&q_pe)?;

        // === KV path ===
        // Compress: [B, S, hidden] → [B, S, kv_lora_rank + rope_head_dim]
        let kv_compressed = self.kv_compress.forward(client, hidden)?;

        // Split: c_kv [B, S, kv_lora_rank], k_pe_raw [B, S, rope_head_dim]
        let c_kv = var_narrow(&kv_compressed, 2, 0, self.kv_lora_rank).map_err(Error::Numr)?;
        let c_kv = var_contiguous(&c_kv)?;
        let k_pe_raw = var_narrow(&kv_compressed, 2, self.kv_lora_rank, self.rope_head_dim)
            .map_err(Error::Numr)?;
        let k_pe_raw = var_contiguous(&k_pe_raw)?;

        // Normalize c_kv
        let c_kv = if let Some(norm) = &self.kv_norm {
            norm.forward(client, &c_kv)?
        } else {
            c_kv
        };

        // Decompress: [B, S, kv_lora_rank] → [B, S, num_heads * (head_dim + head_dim_v)]
        let kv = self.kv_decompress.forward(client, &c_kv)?;
        let kv = var_reshape(
            &kv,
            &[
                batch,
                seq_len,
                self.num_heads,
                self.head_dim + self.head_dim_v,
            ],
        )
        .map_err(Error::Numr)?;
        // → [B, H, S, head_dim + head_dim_v]
        let kv = var_permute(&kv, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let kv = var_contiguous(&kv)?;

        // Split K_nope and V
        let k_nope = var_narrow(&kv, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let k_nope = var_contiguous(&k_nope)?;
        let v = var_narrow(&kv, 3, self.head_dim, self.head_dim_v).map_err(Error::Numr)?;
        let v = var_contiguous(&v)?;

        // K_pe: [B, S, rope_head_dim] → [B, 1, S, rope_head_dim] → [B, H, S, rope_head_dim]
        let k_pe = var_reshape(&k_pe_raw, &[batch, 1, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_broadcast_to(&k_pe, &[batch, self.num_heads, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_contiguous(&k_pe)?;

        // Apply RoPE to q_pe and k_pe (decoupled)
        let q_pe = apply_rope_impl(client, &q_pe, self.rope.cos_cache(), self.rope.sin_cache())?;
        let k_pe = apply_rope_impl(client, &k_pe, self.rope.cos_cache(), self.rope.sin_cache())?;

        // Concatenate: Q = [q_nope, q_pe], K = [k_nope, k_pe]
        let q = var_cat(&[&q_nope, &q_pe], 3, client).map_err(Error::Numr)?;
        let k = var_cat(&[&k_nope, &k_pe], 3, client).map_err(Error::Numr)?;

        // Attention: Q,K [B, H, S, qk_dim], V [B, H, S, head_dim_v]
        let attn_out = scaled_dot_product_attention_impl(client, &q, &k, &v, self.scale, true)?;

        // [B, H, S, head_dim_v] → [B, S, H, head_dim_v] → [B, S, H*head_dim_v]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(
            &attn_out,
            &[batch, seq_len, self.num_heads * self.head_dim_v],
        )
        .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

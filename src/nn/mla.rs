//! Multi-Head Latent Attention (MLA) module
//!
//! DeepSeek-V2 style attention with low-rank KV compression.
//! Compresses KV cache from O(L * n_heads * head_dim * 2) to O(L * (kv_lora_rank + rope_head_dim)).
//!
//! Architecture:
//! - Q path: optional low-rank compression (q_down → norm → q_up) or direct projection
//! - KV path: compress → split (c_kv, k_pe) → norm c_kv → decompress → split (k_nope, v)
//! - Decoupled RoPE: applied only to q_pe and k_pe portions
//! - Attention: Q=[q_nope, q_pe], K=[k_nope, k_pe], V=v

use crate::error::{Error, Result};
use crate::nn::{Linear, RmsNorm, RoPE, VarBuilder};
use crate::ops::RoPEOps;
use crate::ops::impl_generic::mla::scaled_dot_product_attention_impl;
use crate::ops::impl_generic::rope::apply_rope_impl;
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// MLA configuration
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head for Q/K nope portion
    pub head_dim: usize,
    /// Dimension per head for values (can differ from head_dim)
    pub head_dim_v: usize,
    /// KV compression latent dimension
    pub kv_lora_rank: usize,
    /// Q compression latent dimension (0 = no compression)
    pub q_lora_rank: usize,
    /// Decoupled RoPE dimension
    pub rope_head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base theta
    pub rope_theta: f32,
    /// Whether to use RMSNorm on compressed representations
    pub use_norm: bool,
    /// RMSNorm epsilon
    pub norm_eps: f32,
}

impl MlaConfig {
    /// Create config with DeepSeek-V2 defaults
    pub fn deepseek_v2(
        hidden_size: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
        rope_head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            hidden_size,
            num_heads,
            head_dim,
            head_dim_v: head_dim,
            kv_lora_rank,
            q_lora_rank,
            rope_head_dim,
            max_seq_len,
            rope_theta: 10000.0,
            use_norm: true,
            norm_eps: 1e-6,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(Error::ModelError {
                reason: "hidden_size and num_heads must be > 0".into(),
            });
        }
        if self.kv_lora_rank == 0 {
            return Err(Error::ModelError {
                reason: "kv_lora_rank must be > 0 for MLA".into(),
            });
        }
        if self.rope_head_dim > self.head_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_head_dim ({}) > head_dim ({})",
                    self.rope_head_dim, self.head_dim
                ),
            });
        }
        Ok(())
    }

    /// Total Q/K dimension per head (nope + pe)
    pub fn qk_head_dim(&self) -> usize {
        self.head_dim + self.rope_head_dim
    }

    /// Whether Q uses low-rank compression
    pub fn q_uses_lora(&self) -> bool {
        self.q_lora_rank > 0
    }
}

/// Multi-Head Latent Attention (MLA) layer from DeepSeek-V2.
///
/// Implements low-rank KV compression with decoupled RoPE:
/// - **Q path**: optional low-rank compression (`q_down` → norm → `q_up`) or direct projection
/// - **KV path**: `kv_compress` → norm → `kv_decompress` → split into `(k_nope, v)`
/// - **RoPE**: applied separately to `q_pe` and `k_pe` (decoupled from compressed latent)
/// - **Output**: `softmax(Q·K^T / √d) · V` projected through `o_proj`
pub struct Mla<R: Runtime> {
    // Q path
    q_down: Option<Linear<R>>,
    q_up: Linear<R>,
    q_norm: Option<RmsNorm<R>>,

    // KV path
    kv_compress: Linear<R>,
    kv_norm: Option<RmsNorm<R>>,
    kv_decompress: Linear<R>,

    // Output
    o_proj: Linear<R>,

    // RoPE
    rope: RoPE<R>,

    // Config
    num_heads: usize,
    head_dim: usize,
    head_dim_v: usize,
    rope_head_dim: usize,
    kv_lora_rank: usize,
    scale: f64,
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
            let q_down = Linear::new(
                Tensor::<R>::zeros(&[config.q_lora_rank, h], dt, device),
                None,
                true,
            );
            let q_up = Linear::new(
                Tensor::<R>::zeros(&[nh * qk_dim, config.q_lora_rank], dt, device),
                None,
                true,
            );
            let q_norm = if config.use_norm {
                Some(RmsNorm::new(
                    Tensor::<R>::ones(&[config.q_lora_rank], dt, device),
                    config.norm_eps,
                    true,
                ))
            } else {
                None
            };
            (Some(q_down), q_up, q_norm)
        } else {
            let q_up = Linear::new(
                Tensor::<R>::zeros(&[nh * qk_dim, h], dt, device),
                None,
                true,
            );
            (None, q_up, None)
        };

        let kv_compress = Linear::new(
            Tensor::<R>::zeros(&[config.kv_lora_rank + config.rope_head_dim, h], dt, device),
            None,
            true,
        );
        let kv_norm = if config.use_norm {
            Some(RmsNorm::new(
                Tensor::<R>::ones(&[config.kv_lora_rank], dt, device),
                config.norm_eps,
                true,
            ))
        } else {
            None
        };
        let kv_decompress = Linear::new(
            Tensor::<R>::zeros(
                &[
                    nh * (config.head_dim + config.head_dim_v),
                    config.kv_lora_rank,
                ],
                dt,
                device,
            ),
            None,
            true,
        );

        let o_proj = Linear::new(
            Tensor::<R>::zeros(&[h, nh * config.head_dim_v], dt, device),
            None,
            true,
        );

        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            config.rope_head_dim,
            config.rope_theta,
            None,
            device,
        );

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
            let mut qa_vb = vb.pp("q_a_proj");
            let q_down = Linear::new(qa_vb.take_tensor("weight")?, None, false);

            let mut qb_vb = vb.pp("q_b_proj");
            let q_up = Linear::new(qb_vb.take_tensor("weight")?, None, false);

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
            let mut q_vb = vb.pp("q_proj");
            let q_up = Linear::new(q_vb.take_tensor("weight")?, None, false);
            (None, q_up, None)
        };

        // KV path
        let mut kva_vb = vb.pp("kv_a_proj_with_mqa");
        let kv_compress = Linear::new(kva_vb.take_tensor("weight")?, None, false);

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

        let mut kvb_vb = vb.pp("kv_b_proj");
        let kv_decompress = Linear::new(kvb_vb.take_tensor("weight")?, None, false);

        // Output
        let mut o_vb = vb.pp("o_proj");
        let o_proj = Linear::new(o_vb.take_tensor("weight")?, None, false);

        // RoPE
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            config.rope_head_dim,
            config.rope_theta,
            None,
            vb.device(),
        );

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
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
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
        let q = var_contiguous(&q);

        // Split Q into nope and pe
        let q_nope = var_narrow(&q, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let q_nope = var_contiguous(&q_nope);
        let q_pe = var_narrow(&q, 3, self.head_dim, self.rope_head_dim).map_err(Error::Numr)?;
        let q_pe = var_contiguous(&q_pe);

        // === KV path ===
        // Compress: [B, S, hidden] → [B, S, kv_lora_rank + rope_head_dim]
        let kv_compressed = self.kv_compress.forward(client, hidden)?;

        // Split: c_kv [B, S, kv_lora_rank], k_pe_raw [B, S, rope_head_dim]
        let c_kv = var_narrow(&kv_compressed, 2, 0, self.kv_lora_rank).map_err(Error::Numr)?;
        let c_kv = var_contiguous(&c_kv);
        let k_pe_raw = var_narrow(&kv_compressed, 2, self.kv_lora_rank, self.rope_head_dim)
            .map_err(Error::Numr)?;
        let k_pe_raw = var_contiguous(&k_pe_raw);

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
        let kv = var_contiguous(&kv);

        // Split K_nope and V
        let k_nope = var_narrow(&kv, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let k_nope = var_contiguous(&k_nope);
        let v = var_narrow(&kv, 3, self.head_dim, self.head_dim_v).map_err(Error::Numr)?;
        let v = var_contiguous(&v);

        // K_pe: [B, S, rope_head_dim] → [B, 1, S, rope_head_dim] → [B, H, S, rope_head_dim]
        let k_pe = var_reshape(&k_pe_raw, &[batch, 1, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_broadcast_to(&k_pe, &[batch, self.num_heads, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_contiguous(&k_pe);

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
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(
            &attn_out,
            &[batch, seq_len, self.num_heads * self.head_dim_v],
        )
        .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

/// Make a Var contiguous (copies data if non-contiguous layout).
fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

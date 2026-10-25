//! [`AbsorbedMla`]: an [`Mla`] layer in absorbed form, for KV-cached
//! inference, and [`Mla::absorb`] that builds it.
//!
//! The absorbed form runs MLA as multi-query attention over the latent. One
//! cache head per token holds `[c_kv | k_pe]`, `kv_lora_rank + rope_head_dim`
//! wide, and the V half of the cache holds the same entry:
//!
//! - The K half of `kv_b_proj` (`W_UK`, per head `[head_dim, kv_lora_rank]`)
//!   folds into the query: `q_nope · (W_UK c) = (q_nope W_UK) · c`.
//! - The attention output over the latent goes through the V half of
//!   `kv_b_proj` (`W_UV`, per head `[head_dim_v, kv_lora_rank]`) per head,
//!   then through `o_proj` unchanged.
//!
//! The scores and the output equal the unabsorbed [`Mla::forward`] exactly in
//! exact arithmetic. The model never rebuilds per-head K or V for the cache.

use super::model::Mla;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::nn::{MaybeQuantLinear, RmsNorm, RoPE};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// An [`Mla`] layer in absorbed form. See the module docs.
///
/// Build it with [`Mla::absorb`]. Run it with
/// [`AbsorbedMla::forward_inference`] over a [`KvCache`](crate::inference::KvCache)
/// of [`cache_kv_heads`](Self::cache_kv_heads) heads, each
/// [`cache_head_dim`](Self::cache_head_dim) wide.
pub struct AbsorbedMla<R: Runtime> {
    pub(super) q_down: Option<MaybeQuantLinear<R>>,
    pub(super) q_up: MaybeQuantLinear<R>,
    pub(super) q_norm: Option<RmsNorm<R>>,
    pub(super) kv_compress: MaybeQuantLinear<R>,
    pub(super) kv_norm: Option<RmsNorm<R>>,
    /// `W_UK`: `[1, num_heads, head_dim, kv_lora_rank]`.
    pub(super) w_uk: Var<R>,
    /// `W_UV` transposed per head: `[1, num_heads, kv_lora_rank, head_dim_v]`.
    pub(super) w_uv_t: Var<R>,
    pub(super) o_proj: MaybeQuantLinear<R>,
    pub(super) rope: RoPE<R>,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) head_dim_v: usize,
    pub(super) rope_head_dim: usize,
    pub(super) kv_lora_rank: usize,
    pub(super) scale: f64,
}

impl<R: Runtime<DType = DType>> Mla<R> {
    /// This layer in absorbed form, for KV-cached inference.
    ///
    /// `kv_b_proj` splits into its per-head K and V halves. A quantized
    /// `kv_b_proj` dequantizes to `dtype` first. Every other projection moves
    /// over unchanged, quantized or not.
    ///
    /// # Errors
    ///
    /// A `kv_b_proj` with a bias, with a shape other than
    /// `[num_heads * (head_dim + head_dim_v), kv_lora_rank]`, or in the
    /// decomposed quantized layout.
    pub fn absorb(self, dtype: DType) -> Result<AbsorbedMla<R>>
    where
        R::Client: DequantOps<R>,
    {
        let (h, d, dv, r) = (
            self.num_heads,
            self.head_dim,
            self.head_dim_v,
            self.kv_lora_rank,
        );
        let w = dense_weight(&self.kv_decompress, dtype)?;
        if w.shape() != [h * (d + dv), r] {
            return Err(Error::ModelError {
                reason: format!(
                    "MLA kv_b_proj is {:?}; absorption needs [{}, {r}] \
                     (num_heads {h} * (head_dim {d} + head_dim_v {dv}), kv_lora_rank {r})",
                    w.shape(),
                    h * (d + dv)
                ),
            });
        }
        let per_head = w.contiguous()?.reshape(&[h, d + dv, r])?;
        let w_uk = per_head
            .narrow(1, 0, d)?
            .contiguous()?
            .reshape(&[1, h, d, r])?;
        let w_uv_t = per_head
            .narrow(1, d, dv)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape(&[1, h, r, dv])?;
        Ok(AbsorbedMla {
            q_down: self.q_down,
            q_up: self.q_up,
            q_norm: self.q_norm,
            kv_compress: self.kv_compress,
            kv_norm: self.kv_norm,
            w_uk: Var::new(w_uk, false),
            w_uv_t: Var::new(w_uv_t, false),
            o_proj: self.o_proj,
            rope: self.rope,
            num_heads: h,
            head_dim: d,
            head_dim_v: dv,
            rope_head_dim: self.rope_head_dim,
            kv_lora_rank: r,
            scale: self.scale,
        })
    }
}

impl<R: Runtime> AbsorbedMla<R> {
    /// KV heads the cache holds: one, shared by every query head.
    pub fn cache_kv_heads(&self) -> usize {
        1
    }

    /// Width of one cache entry, K and V alike: `kv_lora_rank + rope_head_dim`.
    pub fn cache_head_dim(&self) -> usize {
        self.kv_lora_rank + self.rope_head_dim
    }

    /// Query heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// The decoupled RoPE tables, `[max_seq_len, rope_head_dim / 2]`.
    pub fn rope(&self) -> &RoPE<R> {
        &self.rope
    }
}

impl<R: Runtime<DType = DType>> AbsorbedMla<R> {
    /// `Ok` when `cache` has this layer's head count and width, and a step at
    /// `position` appends at its end.
    pub(super) fn check_cache(&self, cache: &KvCache<R>, position: usize) -> Result<()>
    where
        R::Client: IndexingOps<R>,
    {
        if cache.num_kv_heads() != self.cache_kv_heads()
            || cache.head_dim() != self.cache_head_dim()
        {
            return Err(Error::InferenceError {
                reason: format!(
                    "AbsorbedMla needs a KV cache of {} head of width {}, got {} heads of \
                     width {}",
                    self.cache_kv_heads(),
                    self.cache_head_dim(),
                    cache.num_kv_heads(),
                    cache.head_dim()
                ),
            });
        }
        if position != cache.seq_len() {
            return Err(Error::InferenceError {
                reason: format!(
                    "AbsorbedMla step at position {position}, but the KV cache holds {} \
                     tokens; pass position = cache.seq_len()",
                    cache.seq_len()
                ),
            });
        }
        Ok(())
    }
}

/// The dense weight of `linear`: the tensor itself, or a quantized weight
/// dequantized to `dtype`.
fn dense_weight<R: Runtime<DType = DType>>(
    linear: &MaybeQuantLinear<R>,
    dtype: DType,
) -> Result<Tensor<R>>
where
    R::Client: DequantOps<R>,
{
    match linear {
        MaybeQuantLinear::Standard(l) => {
            if l.bias().is_some() {
                return Err(Error::ModelError {
                    reason: "MLA kv_b_proj has a bias; absorption folds a bias-free \
                             kv_b_proj only"
                        .into(),
                });
            }
            Ok(l.weight().tensor().clone())
        }
        MaybeQuantLinear::Quantized(q) => {
            let client = R::default_client(q.weight().device());
            client.dequantize(q.weight(), dtype)
        }
        MaybeQuantLinear::DecomposedQuant(_) => Err(Error::ModelError {
            reason: "MLA kv_b_proj is in the decomposed quantized layout, which absorption \
                     does not read; store kv_b_proj dense or block-quantized"
                .into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::MlaConfig;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn absorbed() -> AbsorbedMla<CpuRuntime> {
        let config = MlaConfig::deepseek_v2(16, 2, 8, 6, 4, 16);
        Mla::<CpuRuntime>::from_config(&config, &CpuDevice::new())
            .unwrap()
            .absorb(DType::F32)
            .unwrap()
    }

    #[test]
    fn absorb_splits_kv_b_proj_per_head() {
        let mla = absorbed();
        assert_eq!((mla.cache_kv_heads(), mla.cache_head_dim()), (1, 12));
        assert_eq!(mla.num_heads(), 2);
        assert_eq!(mla.w_uk.shape(), &[1, 2, 8, 8]);
        assert_eq!(mla.w_uv_t.shape(), &[1, 2, 8, 8]);
    }

    #[test]
    fn a_mismatched_cache_or_position_is_refused() {
        let dev = CpuDevice::new();
        let mla = absorbed();
        let wrong = KvCache::<CpuRuntime>::new(1, 2, 4, 16, 12, DType::F32, &dev).unwrap();
        let err = mla.check_cache(&wrong, 0).unwrap_err().to_string();
        assert!(err.contains("1 head of width 12"), "{err}");

        let cache = KvCache::<CpuRuntime>::new(1, 1, 4, 16, 12, DType::F32, &dev).unwrap();
        assert!(mla.check_cache(&cache, 0).is_ok());
        let err = mla.check_cache(&cache, 3).unwrap_err().to_string();
        assert!(err.contains("position = cache.seq_len()"), "{err}");
    }
}

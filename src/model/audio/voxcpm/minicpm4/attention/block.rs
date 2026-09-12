//! [`MiniCpm4Attention`]: the projections, the full-sequence causal forward,
//! `alias`, and the `Module` impl.

use super::guards::missing_rope;
use crate::error::{Error, Result};
use crate::model::attention_core::{AttentionCoreSpec, AttentionKernel, attention_core_masked};
use crate::model::traits::ModelClient;
use crate::nn::{MaybeLoraLinear, MaybeQuantLinear, Module, RoPE, child_params, extend_named};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

/// `q_proj`: 2048 -> 2048 (16 heads x 128), `k_proj`/`v_proj`: 2048 -> 256
/// (2 heads x 128, GQA group size 8), `o_proj`: 2048 -> 2048. All bias-free.
///
/// The projections are [`MaybeLoraLinear`], not plain `Linear`: a GGUF
/// checkpoint stores them block-quantized, and this is the bulk of VoxCPM2's
/// weights. The quantized variant multiplies through `quant_matmul` with the
/// weight left PACKED, so a Q4_K file costs Q4_K-sized memory instead of
/// being expanded to dense F32 at load. A safetensors checkpoint yields the
/// `Standard` variant and runs exactly the dense path it always did.
/// `MaybeLoraLinear` additionally lets any of the four carry a LoRA adapter.
pub struct MiniCpm4Attention<R: Runtime> {
    pub(crate) q_proj: MaybeLoraLinear<R>,
    pub(crate) k_proj: MaybeLoraLinear<R>,
    pub(crate) v_proj: MaybeLoraLinear<R>,
    pub(crate) o_proj: MaybeLoraLinear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// NoPE: skip the rotary embedding on BOTH paths (`residual_lm`).
    ///
    /// `false` for `base_lm`, where every code path runs exactly as it did
    /// before this flag existed.
    pub(crate) no_rope: bool,
}

impl<R: Runtime<DType = DType>> MiniCpm4Attention<R> {
    /// Borrowed view of this block's attention parameters, for
    /// [`attention_core_masked`].
    ///
    /// No Q/K per-head norm and no ALiBi on this checkpoint, and
    /// `sliding_window: 0` — MiniCPM4's VoxCPM2 configuration attends the
    /// full prefix, so windowing is disabled rather than left unset.
    ///
    /// `skip_rope` carries `no_rope` through. It is INDEPENDENT of
    /// `use_alibi`, which stays `false` here: ALiBi would add a distance bias
    /// that `residual_lm` does not have.
    pub(super) fn core_spec(&self) -> AttentionCoreSpec<'_, R> {
        AttentionCoreSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_norm: None,
            k_norm: None,
            use_alibi: false,
            skip_rope: self.no_rope,
            sliding_window: 0,
            // Only `forward` reads this field. `attention_core_flash` builds a
            // backward node that resolves its client from the runtime, so it
            // requires `R::Client: FlashAttentionOps<R>` — a bound this block's
            // where-clause does not carry. `forward_cached` is unaffected: it
            // calls `flash_attention_fwd` on `client` (already a
            // `FlashAttentionOps<R>` via `ModelClient<R>`), with no backward.
            kernel: AttentionKernel::Masked,
        }
    }

    /// Dtype and device the K/V this block writes will actually have.
    ///
    /// Read off `k_proj` because it is that projection's OUTPUT that lands in
    /// the cache. A quantized projection cannot answer this from its weight:
    /// `quant_matmul` consumes F32 and emits F32 whatever block format the
    /// weight is packed in, so the cached keys are F32 there — the packed
    /// weight has no element dtype to copy.
    pub(crate) fn kv_dtype_device(&self) -> Result<(DType, &R::Device)> {
        match self.k_proj.base() {
            MaybeQuantLinear::Standard(linear) => {
                let w = linear.weight().tensor();
                Ok((w.dtype(), w.device()))
            }
            MaybeQuantLinear::Quantized(qlinear) => Ok((DType::F32, qlinear.weight().device())),
            MaybeQuantLinear::DecomposedQuant(_) => Err(Error::ModelError {
                reason: "MiniCPM4 k_proj: no VoxCPM2 checkpoint loads decomposed-quantized \
                         (AWQ/GPTQ) weights, so the KV cache dtype is undefined here"
                    .to_string(),
            }),
        }
    }

    /// Causal GQA attention over `x: [batch, seq, hidden]`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Softmax scale is `1/sqrt(head_dim)`, derived from `q`'s actual last
    /// dimension inside `multi_head_attention_impl`.
    ///
    /// `rope` may be `None` only when `no_rope` is set; otherwise it is an
    /// [`Error::InvalidArgument`], because running unrotated would stay
    /// shape-valid while computing a different model.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: Option<&RoPE<R>>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeLoraLinear::forward` adds over a
        // dense `Linear::forward`: its decomposed-quant arm casts activations
        // to F32. `ModelClient` already carries `QuantMatmulOps`.
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        // One activation pass for the three projections: a quantized weight
        // set quantizes `x` once and reuses it.
        let mut qkv =
            MaybeLoraLinear::forward_batch(&[&self.q_proj, &self.k_proj, &self.v_proj], client, x)?
                .into_iter();
        let (Some(q), Some(k), Some(v)) = (qkv.next(), qkv.next(), qkv.next()) else {
            return Err(Error::ModelError {
                reason: "forward_batch returned fewer outputs than layers".into(),
            });
        };

        let (cos, sin) = match (self.no_rope, rope) {
            (true, _) => (None, None),
            (false, Some(rope)) => (Some(rope.cos_cache()), Some(rope.sin_cache())),
            (false, None) => return Err(missing_rope()),
        };

        let attn_out = attention_core_masked(client, &q, &k, &v, cos, sin, &self.core_spec())?;

        self.o_proj.forward(client, &attn_out)
    }

    /// Cheap duplicate that preserves every projection's `Var<R>`
    /// `TensorId`s, for capturing this block by owned value in a `'static`
    /// activation-checkpointing closure — `numr::autograd::checkpoint`'s
    /// closure is `Fn(...) + Send + Sync + 'static`, so a layer cannot be
    /// borrowed into it. Each projection routes through
    /// [`MaybeLoraLinear::alias`], never [`Clone`], so the optimizer, keyed
    /// by `TensorId`, still sees the original parameters' gradients.
    pub fn alias(&self) -> Self {
        Self {
            q_proj: self.q_proj.alias(),
            k_proj: self.k_proj.alias(),
            v_proj: self.v_proj.alias(),
            o_proj: self.o_proj.alias(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            no_rope: self.no_rope,
        }
    }
}

/// Names ARE the field names (`q_proj`, `k_proj`, `v_proj`, `o_proj`) —
/// the `self_attn` checkpoint segment is added by the owning
/// [`MiniCpm4Layer`](crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer).
/// `no_rope` carries no `Var<R>` (it is a `bool`), so it is correctly absent
/// from every collection below.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Attention<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.q_proj);
        params.extend(child_params(&self.k_proj));
        params.extend(child_params(&self.v_proj));
        params.extend(child_params(&self.o_proj));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "q_proj", self.q_proj.named_parameters());
        extend_named(&mut params, "k_proj", self.k_proj.named_parameters());
        extend_named(&mut params, "v_proj", self.v_proj.named_parameters());
        extend_named(&mut params, "o_proj", self.o_proj.named_parameters());
        params
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::nn::Weight;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    pub(in super::super) const HIDDEN: usize = 4;
    pub(in super::super) const NUM_HEADS: usize = 1;
    pub(in super::super) const NUM_KV_HEADS: usize = 1;
    pub(in super::super) const HEAD_DIM: usize = 4;

    /// Deterministic, non-degenerate weights: zeros would make every
    /// assertion below pass vacuously.
    fn filled(shape: &[usize], salt: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 29 + salt * 7) % 11) as f32 - 5.0) / 8.0)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("weights")
    }

    pub(in super::super) fn tiny_attention(
        no_rope: bool,
        device: &CpuDevice,
    ) -> MiniCpm4Attention<CpuRuntime> {
        let linear = |salt| -> MaybeLoraLinear<CpuRuntime> {
            MaybeQuantLinear::from_weight(
                Weight::Standard(filled(&[HIDDEN, HIDDEN], salt, device)),
                None,
            )
            .into()
        };
        MiniCpm4Attention {
            q_proj: linear(1),
            k_proj: linear(2),
            v_proj: linear(3),
            o_proj: linear(4),
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            no_rope,
        }
    }

    /// One `[1, 1, HIDDEN]` embedding.
    pub(in super::super) fn embed(salt: usize, device: &CpuDevice) -> Var<CpuRuntime> {
        Var::new(filled(&[1, 1, HIDDEN], salt, device), false)
    }

    /// A NoPE block runs to completion with no table at all.
    #[test]
    fn nope_block_runs_without_a_table() {
        let (client, device) = cpu_setup();
        let attn = tiny_attention(true, &device);
        let out = attn
            .forward(&client, &embed(1, &device), None)
            .expect("forward");
        assert_eq!(out.shape(), &[1, 1, HIDDEN]);
    }
}

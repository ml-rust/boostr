//! `forward_with_paged_kv_cache`: QKV projection, Q/K norms, RoPE, the
//! paged cache write and attention, and the output projection.

use super::super::super::helpers::var_contiguous;
use super::super::LlamaAttention;
use super::attend::PagedQkv;
use super::packed_rope::PackedRopeShape;
use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::ops::traits::{KvCacheOps, KvSeqLen, PagedAttentionOps, RoPEPackedOps};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    /// `seq_len_k` is the batch-shared bound used to size the gathered KV
    /// buffer. `kv_seq_len` narrows which of those gathered keys each row
    /// attends over: `None` keeps every row at `seq_len_k`.
    /// `Some(KvSeqLen::PerRow(ends))` lets the caller give a mixed-length
    /// batch its own end per row, so a shorter row does not read into a
    /// longer row's cached blocks.
    ///
    /// `rope_positions` gives each row of the batch its own RoPE position,
    /// replacing the single shared `position` scalar for that row's query.
    /// Shape `[batch * seq_len]` (row-major, matching `x`'s `[B, S]`
    /// layout), I32 or I64. `None` keeps today's behavior: every row uses
    /// the shared `position..position+seq_len` window. Only honored when
    /// `seq_len == 1` (one query token per row, the decode-batch shape) —
    /// a `seq_len > 1` batch cannot derive a position per token from a
    /// per-row tensor and is refused. ALiBi models ignore `rope_positions`
    /// (they carry no RoPE cache) exactly as they ignore `position` today.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        layer_idx: usize,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        position: usize,
        kv_seq_len: Option<KvSeqLen<'_, R>>,
        rope_positions: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + RoPEPackedOps<R>,
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
            + KvCacheOps<R>
            + PagedAttentionOps<R>
            + DequantOps<R>,
    {
        if let Some(feature) = self.paged_unsupported(seq_len_k) {
            return Err(Error::ModelError {
                reason: format!(
                    "the paged-attention path does not apply {feature}; serve this model \
                     with paged attention off"
                ),
            });
        }
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        let qkv = self.project_qkv(client, x)?;
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        // Reshape to [B, S, H, D] then permute to [B, H, S, D]
        let q = var_reshape(q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;
        let v = var_contiguous(&v)?;

        // Optional Q/K layer norms (Command-R, Cohere) — applied before RoPE
        let (q, k) = self.apply_qk_norms(client, &q, &k)?;

        // Apply RoPE or skip for ALiBi models. A per-row `rope_positions`
        // tensor takes the packed (position-id-aware) path so a batch with
        // mixed cache lengths gives each row its own RoPE angle instead of
        // sharing the batch-max `position`. Without it, every row keeps
        // today's single shared cos/sin window.
        let (q, k) = if self.use_alibi {
            (q, k)
        } else if let Some(positions) = rope_positions {
            self.apply_rotary_packed_paged(
                client,
                q,
                k,
                rope,
                positions,
                PackedRopeShape { batch, seq_len },
            )?
        } else {
            let cos_offset =
                var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            let sin_offset =
                var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            self.apply_rotary_if_needed(client, q, k, &cos_offset, &sin_offset)?
        };

        let rc = R::default_client(x.tensor().device());
        let attn_out = self.paged_attend(
            &rc,
            PagedQkv {
                q: &q,
                k: &k,
                v: &v,
                batch,
                seq_len,
            },
            paged_cache,
            layer_idx,
            slot_mapping,
            block_table,
            seq_len_k,
            kv_seq_len,
        )?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

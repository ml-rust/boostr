//! [`MiniCpm4Attention::forward_cached`]: the KV-cached causal path.

use super::block::MiniCpm4Attention;
use super::guards::{missing_rope, require_preallocated_cache};
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::var_ops::var_contiguous;
use crate::nn::{MaybeLoraLinear, RoPE};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> MiniCpm4Attention<R> {
    /// KV-cached causal GQA attention over `x: [batch, seq, hidden]` covering
    /// absolute positions `position..position + seq`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Serves BOTH cached shapes: prefill passes the whole prefix at
    /// `position == 0`, a decode step passes `seq == 1` at the next position.
    ///
    /// This does not call [`attention_core_masked`]: that entry point rotates
    /// the RAW K it is given, so handing it the cache would re-apply RoPE to
    /// keys already rotated when they were written.
    /// The prologue is otherwise the same one that function documents —
    /// reshape/permute, contiguous Q/K, (no Q/K norm here), RoPE. The attention
    /// is the flash kernel, reading the cache in place and masking internally:
    /// nothing here materializes a mask, repeats KV heads, or copies history.
    ///
    /// # Masking
    ///
    /// `causal` is `seq > 1`. A prefill chunk needs it, because each query row
    /// must reject the rows after it. A decode step (`seq == 1`) does not: its
    /// one query row is the last position, so a causal mask would be all zeros.
    ///
    /// The admitted key set is the reference's. `build_attention_mask` in
    /// `ops::impl_generic::attention::flash_standard` offsets keys by
    /// `seq_len_k - seq_len_q` — the offset the shared `prefill_attention_mask`
    /// builder used — so query row `i` admits exactly keys `0..=position + i`.
    /// The reference instead masks with `arange(max_length) <= position_id`
    /// across the FULL preallocated width: same keys, plus `-inf` on zeroed
    /// never-written slots that contribute nothing to the softmax either way,
    /// and `kv_seq_len` stops the kernel reading those slots at all. The
    /// equivalence holds ONLY because the caller writes positions in order from
    /// 0, which [`MiniCpm4Model::decode_step`] enforces by requiring
    /// `position == kv_cache.seq_len()`.
    ///
    /// [`attention_core_masked`]: crate::model::attention_core::attention_core_masked
    /// [`MiniCpm4Model::decode_step`]:
    ///     crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::decode_step
    /// `rope` may be `None` only when `no_rope` is set; otherwise it is an
    /// [`Error::InvalidArgument`]. The decode path never dereferences an
    /// absent table.
    pub fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: Option<&RoPE<R>>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        // `TypeConversionOps` for the same reason `forward` needs it.
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
        let shape = x.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected 3D [batch, seq, hidden], got {}D", shape.len()),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);

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

        // [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        let q =
            var_reshape(&q, &[batch, seq, self.num_heads, self.head_dim]).map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // The fused RoPE kernel assumes contiguous layout.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;

        // Same precomputed cos/sin tables the full-sequence path uses, sliced
        // at the absolute positions this call covers. Building a second table
        // here is how the two paths would drift.
        //
        // NoPE skips the whole block: this is the SECOND site that applies
        // RoPE (the full-sequence path applies it inside
        // `attention_core_masked`), and honouring `no_rope` in only one of
        // them would leave the two paths computing different models.
        let (q, k) = match (self.no_rope, rope) {
            (true, _) => (q, k),
            (false, Some(rope)) => {
                // `position` rotates the query; `kv_cache` stores it at its own
                // `seq_len`. They MUST agree, or the query is rotated for one
                // absolute position and attended at another — shapes stay
                // valid and the output is silently wrong.
                //
                // Checked here, not at the top, because this is the ONLY site
                // that reads `position`. A NoPE stack never rotates, so its
                // `position` is genuinely unused and stays unconstrained —
                // which is what
                // `nope_output_is_independent_of_absolute_position` asserts by
                // passing a `position` a cache-aligned call could not use.
                let written = kv_cache.seq_len();
                if position != written {
                    return Err(Error::InferenceError {
                        reason: format!(
                            "RoPE position {position} disagrees with the KV cache's next \
                             write index {written}: the query would be rotated for position \
                             {position} but stored and attended at {written}. Pass \
                             position == kv_cache.seq_len(), as MiniCpm4Model::decode_step \
                             does."
                        ),
                    });
                }
                let cos = var_narrow(rope.cos_cache(), 0, position, seq).map_err(Error::Numr)?;
                let sin = var_narrow(rope.sin_cache(), 0, position, seq).map_err(Error::Numr)?;
                let q = client.apply_rope(&q, &cos, &sin)?;
                let k = client.apply_rope(&k, &cos, &sin)?;
                (q, k)
            }
            (false, None) => return Err(missing_rope()),
        };

        let v = var_contiguous(&v)?;

        // Post-RoPE K/V land in the cache, as the reference writes rotated keys
        // at index `position_id`. `update_fused` writes the new slots IN PLACE
        // through `KvCacheOps::kv_cache_update`; `update` would `slice_assign`,
        // which is functional — a second buffer of the whole preallocated
        // cache, allocated and copied per layer per step.
        require_preallocated_cache(kv_cache, seq)?;
        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;

        // The kernel broadcasts the GQA heads itself, so the raw cache buffers
        // go in untouched, bounded to the written slots by `kv_seq_len`.
        let (out, _lse) = client.flash_attention_fwd(
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            // Prefill chunk masks; a single decode row has nothing to reject.
            seq > 1,
            // The disabled-window sentinel `core_spec` declares.
            self.core_spec().sliding_window,
            Some(kv_cache.seq_len()),
        )?;
        let attn_out = Var::new(out, false);

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::block::tests::{HEAD_DIM, NUM_KV_HEADS, embed, tiny_attention};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// The load-bearing property of NoPE: the block carries NO positional
    /// signal, so the same embedding attending the same key set produces the
    /// same output whatever absolute position it claims.
    ///
    /// Both runs write one prior key/value into the cache, then present the
    /// SAME query embedding at a different absolute position. The key set and
    /// the causal mask are identical across the two runs (the mask is built
    /// from the cache length, not from `position`), so the rotation is the
    /// only thing that can differ — which is why the rotary half of this test
    /// must, and does, disagree.
    #[test]
    fn nope_output_is_independent_of_absolute_position() {
        let (client, device) = cpu_setup();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, &device)
            .expect("rope");

        let run = |no_rope: bool, position: usize| {
            let attn = tiny_attention(no_rope, &device);
            let table = (!no_rope).then_some(&rope);
            let mut cache =
                KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
                    .expect("cache");
            attn.forward_cached(&client, &embed(1, &device), table, &mut cache, 0)
                .expect("prior position");
            let out = attn
                .forward_cached(&client, &embed(2, &device), table, &mut cache, position)
                .expect("query position");
            out.tensor()
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>()
        };

        let near = run(true, 1);
        let far = run(true, 9);
        assert!(
            near.iter().any(|v| v.abs() > 1e-6),
            "degenerate output: the comparison below would pass vacuously"
        );
        for (a, b) in near.iter().zip(&far) {
            assert!(
                (a - b).abs() < 1e-6,
                "no_rope leaked a positional signal: {a} vs {b}"
            );
        }

        // The NoPE half above is only meaningful if a rotating block is NOT
        // position-invariant. Proving that by re-querying at a mismatched
        // `position` is no longer possible — and should not be: a rotating block
        // now rejects a `position` that disagrees with the cache's next write
        // index, because the query would be rotated for one absolute position and
        // attended at another. So compare rotating against NoPE at the SAME valid
        // position instead: identical weights, identical keys, identical mask, the
        // rotation the only difference.
        let rotary_valid = run(false, 1);
        assert!(
            rotary_valid
                .iter()
                .zip(&near)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "rotating and NoPE agreed at the same position, so rotation was never \
             applied and the NoPE half of this test proves nothing"
        );
    }

    /// A rotating block rejects a `position` that disagrees with the cache's
    /// next write index.
    ///
    /// The query would be rotated for `position` and then stored and attended at
    /// `kv_cache.seq_len()`. Every shape stays valid, so without this the output
    /// is silently wrong. `MiniCpm4Model::decode_step` keeps the two in step; a
    /// direct caller can not be trusted to.
    #[test]
    fn rotating_block_rejects_a_position_the_cache_disagrees_with() {
        let (client, device) = cpu_setup();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, &device)
            .expect("rope");
        let attn = tiny_attention(false, &device);
        let mut cache =
            KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
                .expect("cache");

        attn.forward_cached(&client, &embed(1, &device), Some(&rope), &mut cache, 0)
            .expect("position 0 matches an empty cache");
        assert_eq!(cache.seq_len(), 1, "one slot written");

        let err = attn
            .forward_cached(&client, &embed(2, &device), Some(&rope), &mut cache, 9)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains('9'), "error must name the position: {msg}");
        assert!(msg.contains("seq_len"), "error must name the fix: {msg}");

        // A NoPE block never rotates, so its `position` is unused and unchecked.
        let nope = tiny_attention(true, &device);
        let mut nope_cache =
            KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
                .expect("cache");
        nope.forward_cached(&client, &embed(1, &device), None, &mut nope_cache, 0)
            .expect("prior");
        nope.forward_cached(&client, &embed(2, &device), None, &mut nope_cache, 9)
            .expect("NoPE ignores position, so a mismatch is not an error");
    }

    /// A block that rotates must not fall back to an unrotated forward when
    /// the table is missing. Both paths error, neither panics.
    #[test]
    fn rotating_block_rejects_a_missing_rope_table() {
        let (client, device) = cpu_setup();
        let attn = tiny_attention(false, &device);

        let err = attn.forward(&client, &embed(1, &device), None).unwrap_err();
        assert!(err.to_string().contains("no_rope"), "got {err}");

        let mut cache =
            KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
                .expect("cache");
        let err = attn
            .forward_cached(&client, &embed(1, &device), None, &mut cache, 0)
            .unwrap_err();
        assert!(err.to_string().contains("no_rope"), "got {err}");
        assert_eq!(cache.seq_len(), 0, "cache was written on the error path");
    }
}

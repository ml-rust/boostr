//! Paged cache write and paged attention: writes this step's K/V through
//! the slot mapping, then attends Q over the cached blocks.

use super::super::super::helpers::var_contiguous;
use super::super::LlamaAttention;
use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::ops::traits::{KvCacheOps, KvSeqLen, PagedAttentionOps};
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Post-RoPE Q/K/V, each `[B, H, S, D]` contiguous, with the batch
/// geometry they were shaped from.
pub(super) struct PagedQkv<'a, R: Runtime> {
    pub(super) q: &'a Var<R>,
    pub(super) k: &'a Var<R>,
    pub(super) v: &'a Var<R>,
    pub(super) batch: usize,
    pub(super) seq_len: usize,
}

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    /// Writes K/V into layer `layer_idx` of `paged_cache` through
    /// `slot_mapping`, then runs paged attention of Q against the cached
    /// K/V. Returns `[B, S, H*D]`, ready for the output projection.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paged_attend(
        &self,
        rc: &R::Client,
        qkv: PagedQkv<'_, R>,
        paged_cache: &LayeredPagedKvCache<R>,
        layer_idx: usize,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        kv_seq_len: Option<KvSeqLen<'_, R>>,
    ) -> Result<Var<R>>
    where
        R::Client: KvCacheOps<R> + PagedAttentionOps<R>,
    {
        let PagedQkv {
            q,
            k,
            v,
            batch,
            seq_len,
        } = qkv;

        // Reshape K/V from [B, H_kv, S, D] to [B*S, H_kv, D] for paged cache update
        let k_flat = k
            .tensor()
            .permute(&[0, 2, 1, 3])? // [B, S, H_kv, D]
            .contiguous()?
            .reshape(&[batch * seq_len, self.num_kv_heads, self.head_dim])?;
        let v_flat = v
            .tensor()
            .permute(&[0, 2, 1, 3])? // [B, S, H_kv, D]
            .contiguous()?
            .reshape(&[batch * seq_len, self.num_kv_heads, self.head_dim])?;

        // Write K/V into paged cache using slot_mapping
        let layer_cache = paged_cache.layer(layer_idx);
        layer_cache.update(&k_flat, &v_flat, slot_mapping, rc)?;

        // Paged attention: Q against full cached K/V
        let block_size = paged_cache.block_size();
        let max_num_blocks = block_table.shape()[block_table.shape().len() - 1];
        let required_blocks = seq_len_k.div_ceil(block_size);
        if max_num_blocks < required_blocks {
            return Err(Error::InvalidArgument {
                arg: "block_table",
                reason: format!(
                    "block table width {max_num_blocks} cannot cover seq_len_k={seq_len_k} \
                     with block_size={block_size}; pass a block table with at least \
                     {required_blocks} blocks per row"
                ),
            });
        }
        // Causal for a multi-token chunk: query i must not see key j when
        // j > (seq_len_k - seq_len) + i. A single-token decode step is
        // unaffected either way, since it has no future key to mask.
        let is_prefill = seq_len > 1;
        let (attn_out, _lse) = rc.paged_attention_fwd(
            q.tensor(),
            layer_cache.k_cache(),
            layer_cache.v_cache(),
            block_table,
            self.num_heads,
            self.num_kv_heads,
            seq_len,
            seq_len_k,
            self.head_dim,
            block_size,
            is_prefill,
            kv_seq_len,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        BLOCK_SIZE, HIDDEN, decode_input, dense_reference_output, prefill_cache, prefill_input,
        prefill_slots, slot_mapping_and_block_table, test_attention, test_cache, test_rope,
    };
    use numr::autograd::{Var, var_narrow};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// A full-prompt paged prefill from an empty cache must match the dense
    /// contiguous-cache path (`forward_with_kv_cache`) given the same
    /// weights, RoPE cache, and input. The two paths use unrelated kernels
    /// (`paged_attention_fwd` vs `flash_attention_fwd`) and cache layouts,
    /// so agreement here pins down the causal masking rule itself, not a
    /// shared implementation. Catches a non-causal (or wrongly-causal)
    /// prefill mask.
    #[test]
    fn prefill_matches_dense_causal_reference() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let seq_len = 5;
        let x = prefill_input(seq_len, &device);

        let num_blocks = 2; // ceil(seq_len=5 / BLOCK_SIZE=4)
        let cache = prefill_cache(num_blocks, &device);
        let (slot_mapping, block_table) = prefill_slots(0, seq_len, num_blocks, &device);
        let paged_out = attn
            .forward_with_paged_kv_cache(
                &client,
                &x,
                &rope,
                &cache,
                0,
                &slot_mapping,
                &block_table,
                seq_len,
                0,
                None,
                None,
            )
            .expect("paged prefill must succeed")
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        let dense_out = dense_reference_output(&client, &attn, &rope, &x, seq_len, &device);

        assert_eq!(paged_out.len(), dense_out.len());
        for (i, (&a, &b)) in paged_out.iter().zip(dense_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "prefill vs dense-reference mismatch at {i}: paged={a}, dense={b}"
            );
        }
    }

    /// Feeds a prompt through `forward_with_paged_kv_cache` one token at a
    /// time (`seq_len == 1` on every call, i.e. a chunk-of-1 prefill) and
    /// requires the result to match a one-shot dense causal reference at
    /// each position. A `seq_len == 1` call takes the paged decode kernel
    /// path, not the tiled prefill kernel every `seq_len > 1` chunk takes,
    /// and its `causal` flag (`is_prefill = seq_len > 1`) is always false.
    /// This pins that a chunk-of-1 prefill still agrees with a causal
    /// reference even though it never sets `causal`.
    #[test]
    fn prefill_one_token_at_a_time_matches_dense_causal_reference() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let seq_len = 5;
        let x = prefill_input(seq_len, &device);

        let dense_out = dense_reference_output(&client, &attn, &rope, &x, seq_len, &device);
        let out_dim = dense_out.len() / seq_len;

        let num_blocks = 2; // ceil(seq_len=5 / BLOCK_SIZE=4)
        let cache = prefill_cache(num_blocks, &device);

        for pos in 0..seq_len {
            let x_tok = var_narrow(&x, 1, pos, 1).unwrap();
            let (slot, bt) = prefill_slots(pos, 1, num_blocks, &device);
            let tok_out = attn
                .forward_with_paged_kv_cache(
                    &client,
                    &x_tok,
                    &rope,
                    &cache,
                    0,
                    &slot,
                    &bt,
                    pos + 1,
                    pos,
                    None,
                    None,
                )
                .expect("single-token prefill step must succeed")
                .tensor()
                .contiguous()
                .unwrap()
                .to_vec::<f32>();

            let expected = &dense_out[pos * out_dim..(pos + 1) * out_dim];
            assert_eq!(tok_out.len(), expected.len());
            for (i, (&a, &b)) in tok_out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "position {pos} dim {i} mismatch: one_token={a}, dense={b}"
                );
            }
        }
    }

    /// Two independent checks on a `P`-token prefill followed by an
    /// `S`-token chunk:
    ///
    /// 1. Chunked-vs-one-shot: the second chunk's output must match a
    ///    single one-shot prefill of all `P + S` tokens, for the `S`
    ///    positions the chunk covers. This pins the offset/continuation
    ///    property (right key range, right slot/position arithmetic) but
    ///    both sides run the same paged kernel, so a masking bug shared by
    ///    both sides cancels out and this check alone cannot catch it.
    /// 2. Chunked-vs-dense-reference: the second chunk's output must also
    ///    match the dense contiguous-cache path (`forward_with_kv_cache`)
    ///    run once over all `P + S` tokens. That path uses an unrelated
    ///    kernel and cache layout and does not depend on the paged path's
    ///    `is_prefill` flag, so this check does catch a non-causal (or
    ///    wrongly-causal) paged chunk mask.
    #[test]
    fn chunked_prefill_matches_one_shot_prefill() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let p = 3usize;
        let s = 2usize;
        let total = p + s;
        let num_blocks = 2; // ceil(total=5 / BLOCK_SIZE=4)
        let x = prefill_input(total, &device);
        let x_chunk1 = var_narrow(&x, 1, 0, p).unwrap();
        let x_chunk2 = var_narrow(&x, 1, p, s).unwrap();

        // Chunked: write P tokens, then S more tokens into the same cache.
        let cache = prefill_cache(num_blocks, &device);
        let (slot1, bt1) = prefill_slots(0, p, num_blocks, &device);
        attn.forward_with_paged_kv_cache(
            &client, &x_chunk1, &rope, &cache, 0, &slot1, &bt1, p, 0, None, None,
        )
        .expect("first prefill chunk must succeed");

        let (slot2, bt2) = prefill_slots(p, s, num_blocks, &device);
        let chunk2_out = attn
            .forward_with_paged_kv_cache(
                &client, &x_chunk2, &rope, &cache, 0, &slot2, &bt2, total, p, None, None,
            )
            .expect("second prefill chunk must succeed")
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        // One-shot: the same P + S tokens prefilled in a single call.
        let one_shot_cache = prefill_cache(num_blocks, &device);
        let (slot_all, bt_all) = prefill_slots(0, total, num_blocks, &device);
        let one_shot_out = attn
            .forward_with_paged_kv_cache(
                &client,
                &x,
                &rope,
                &one_shot_cache,
                0,
                &slot_all,
                &bt_all,
                total,
                0,
                None,
                None,
            )
            .expect("one-shot prefill must succeed")
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        let out_dim = chunk2_out.len() / s;
        let shared = &one_shot_out[p * out_dim..total * out_dim];
        assert_eq!(chunk2_out.len(), shared.len());
        for (i, (&a, &b)) in chunk2_out.iter().zip(shared.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "chunked vs one-shot prefill mismatch at {i}: chunked={a}, one_shot={b}"
            );
        }

        // Independent reference: the same `P + S` tokens run through the
        // dense contiguous-cache path (`forward_with_kv_cache`), a
        // different kernel and cache layout from `paged_attention_fwd`
        // that does not read `is_prefill`. Agreement here, unlike the
        // chunked-vs-one-shot check above, cannot pass by both sides
        // sharing the same masking bug.
        let dense_out = dense_reference_output(&client, &attn, &rope, &x, total, &device);
        let dense_shared = &dense_out[p * out_dim..total * out_dim];
        assert_eq!(chunk2_out.len(), dense_shared.len());
        for (i, (&a, &b)) in chunk2_out.iter().zip(dense_shared.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "chunked prefill vs dense-reference mismatch at {i}: chunked={a}, dense={b}"
            );
        }
    }

    /// Regression guard for the decode path: a decode step (`seq_len == 1`)
    /// after a prefill must match a one-shot prefill's last row. This does
    /// NOT detect a causal-masking defect — with one query row, causal
    /// masking removes nothing either way, so `is_prefill` is true or false
    /// for exactly the same output. Masking correctness is covered by
    /// `prefill_matches_dense_causal_reference` and the dense-reference
    /// check in `chunked_prefill_matches_one_shot_prefill`.
    #[test]
    fn decode_after_prefill_is_unaffected_by_causal_flag() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let p = 3usize;
        let num_blocks = 2;
        let x_prefill = prefill_input(p, &device);

        let cache = prefill_cache(num_blocks, &device);
        let (slot1, bt1) = prefill_slots(0, p, num_blocks, &device);
        attn.forward_with_paged_kv_cache(
            &client, &x_prefill, &rope, &cache, 0, &slot1, &bt1, p, 0, None, None,
        )
        .expect("prefill before decode must succeed");

        let decode_data: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.53).sin()).collect();
        let x_decode = Var::new(
            Tensor::<CpuRuntime>::from_slice(&decode_data, &[1, 1, HIDDEN], &device).unwrap(),
            false,
        );
        let (slot2, bt2) = prefill_slots(p, 1, num_blocks, &device);
        let decode_out = attn
            .forward_with_paged_kv_cache(
                &client,
                &x_decode,
                &rope,
                &cache,
                0,
                &slot2,
                &bt2,
                p + 1,
                p,
                None,
                None,
            )
            .expect("decode step must succeed")
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        // Reference: decode is a `seq_len == 1` query against the same
        // cached keys, which `paged_attention_fwd` masks identically
        // whether `causal` is true or false (one row, no future key to
        // drop). A one-shot prefill of all `p + 1` tokens gives the exact
        // same last-row output, since query `p` sees the identical key
        // range `[0, p]` either way.
        let one_shot_cache = prefill_cache(num_blocks, &device);
        let x_all_data: Vec<f32> = {
            let mut d: Vec<f32> = (0..p * HIDDEN).map(|i| (i as f32 * 0.11).cos()).collect();
            d.extend(decode_data.iter().copied());
            d
        };
        let x_all = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_all_data, &[1, p + 1, HIDDEN], &device).unwrap(),
            false,
        );
        let (slot_all, bt_all) = prefill_slots(0, p + 1, num_blocks, &device);
        let one_shot_out = attn
            .forward_with_paged_kv_cache(
                &client,
                &x_all,
                &rope,
                &one_shot_cache,
                0,
                &slot_all,
                &bt_all,
                p + 1,
                0,
                None,
                None,
            )
            .expect("one-shot reference prefill must succeed")
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        let out_dim = decode_out.len();
        let last_row = &one_shot_out[p * out_dim..(p + 1) * out_dim];
        for (i, (&a, &b)) in decode_out.iter().zip(last_row.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "decode-after-prefill mismatch at {i}: decode={a}, one_shot_last_row={b}"
            );
        }
    }

    /// A block table with fewer columns than `seq_len_k` needs (given
    /// `block_size`) must be refused before any gather runs, not silently
    /// read past its row into another row's block-table entries.
    #[test]
    fn block_table_too_narrow_for_seq_len_k_is_rejected() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let batch = 1;
        let cache = test_cache(batch, &device);
        let (slot_mapping, block_table) = slot_mapping_and_block_table(batch, &device);
        let x = decode_input(batch, &device);

        // block_table has one column (one block per row); this seq_len_k
        // needs three blocks at BLOCK_SIZE=4.
        let seq_len_k = BLOCK_SIZE * 2 + 1;

        let result = attn.forward_with_paged_kv_cache(
            &client,
            &x,
            &rope,
            &cache,
            0,
            &slot_mapping,
            &block_table,
            seq_len_k,
            0,
            None,
            None,
        );
        assert!(
            result.is_err(),
            "a block table narrower than seq_len_k / block_size must be refused"
        );
    }
}

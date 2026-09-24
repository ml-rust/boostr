//! Paged-attention forward — decode against a `LayeredPagedKvCache`.

use super::super::helpers::var_contiguous;
use super::LlamaAttention;
use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RoPE};
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

/// Batch geometry for a packed-rope call: `batch * seq_len` gives the
/// packed row count `apply_rope_packed` expects.
#[derive(Clone, Copy)]
struct PackedRopeShape {
    batch: usize,
    seq_len: usize,
}

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
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V projections
        let qkv = MaybeQuantLinear::forward_batch(
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            client,
            x,
        )?;
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
        let rc = R::default_client(x.tensor().device());
        layer_cache.update(&k_flat, &v_flat, slot_mapping, &rc)?;

        // Paged attention: Q against full cached K/V
        let block_size = paged_cache.block_size();
        let _max_num_blocks = block_table.shape()[block_table.shape().len() - 1];
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
            false, // not causal for decode (Q sees all of KV cache)
            kv_seq_len,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

    /// Per-row RoPE for a paged decode batch, via `RoPEPackedOps`.
    ///
    /// `q`/`k` are `[B, H, S, D]` contiguous. Reshaping straight to the
    /// packed `[B*S, H, D]` layout `apply_rope_packed` wants would silently
    /// reorder elements, since a `[B, H, S, D]` buffer is laid out
    /// head-major, not token-major — so this permutes to `[B, S, H, D]`,
    /// forces that layout contiguous, and only then reshapes to
    /// `[B*S, H, D]`, undoing the same steps on the way back.
    ///
    /// Only `seq_len == 1` is honored: `positions` carries one id per row,
    /// which only maps onto `total_tokens = batch * seq_len` when each row
    /// contributes a single query token.
    fn apply_rotary_packed_paged<C>(
        &self,
        client: &C,
        q: Var<R>,
        k: Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
        shape: PackedRopeShape,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: RoPEPackedOps<R>,
    {
        let PackedRopeShape { batch, seq_len } = shape;
        if seq_len != 1 {
            return Err(Error::InvalidArgument {
                arg: "rope_positions",
                reason: format!(
                    "per-row rope positions require exactly one query token per row, \
                     got seq_len={seq_len} for batch={batch}; pass rope_positions=None \
                     and rely on the shared `position` scalar for a multi-token row"
                ),
            });
        }

        let total_tokens = batch * seq_len;
        let pos_shape = positions.shape();
        if pos_shape.len() != 1 || pos_shape[0] != total_tokens {
            return Err(Error::InvalidArgument {
                arg: "rope_positions",
                reason: format!(
                    "expected shape [{total_tokens}] (batch={batch} * seq_len={seq_len}), \
                     got {pos_shape:?}"
                ),
            });
        }

        let q_packed = Self::to_packed(&q, batch, seq_len, self.num_heads, self.head_dim)?;
        let k_packed = Self::to_packed(&k, batch, seq_len, self.num_kv_heads, self.head_dim)?;

        let q_packed =
            client.apply_rope_packed(&q_packed, rope.cos_cache(), rope.sin_cache(), positions)?;
        let k_packed =
            client.apply_rope_packed(&k_packed, rope.cos_cache(), rope.sin_cache(), positions)?;

        let q = Self::from_packed(&q_packed, batch, seq_len, self.num_heads, self.head_dim)?;
        let k = Self::from_packed(&k_packed, batch, seq_len, self.num_kv_heads, self.head_dim)?;
        Ok((q, k))
    }

    /// `[B, H, S, D]` contiguous -> `[B*S, H, D]` packed, via `[B, S, H, D]`.
    fn to_packed(
        x: &Var<R>,
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Var<R>> {
        let x = numr::autograd::var_permute(x, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let x = var_contiguous(&x)?;
        var_reshape(&x, &[batch * seq_len, heads, head_dim]).map_err(Error::Numr)
    }

    /// `[B*S, H, D]` packed -> `[B, H, S, D]` contiguous, undoing `to_packed`.
    fn from_packed(
        x: &Var<R>,
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Var<R>> {
        let x = var_reshape(x, &[batch, seq_len, heads, head_dim]).map_err(Error::Numr)?;
        let x = numr::autograd::var_permute(&x, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        var_contiguous(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayeredPagedKvCache;
    use crate::nn::{Linear, MaybeQuantLinear, RoPE};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const HIDDEN: usize = 8;
    const NUM_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const BLOCK_SIZE: usize = 4;

    /// `[n, n]` identity, so `Linear::forward` is a pass-through and Q/K/V
    /// equal the (reshaped) input — the test then only exercises RoPE and
    /// the paged-cache plumbing, not arbitrary projection weights.
    fn identity_weight(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::<CpuRuntime>::from_slice(&data, &[n, n], device).unwrap()
    }

    fn test_attention(device: &CpuDevice) -> LlamaAttention<CpuRuntime> {
        let identity = || {
            MaybeQuantLinear::Standard(Linear::new(identity_weight(HIDDEN, device), None, false))
        };
        LlamaAttention {
            q_proj: identity(),
            k_proj: identity(),
            v_proj: identity(),
            o_proj: identity(),
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            q_norm: None,
            k_norm: None,
            use_alibi: false,
            sliding_window: 0,
        }
    }

    fn test_rope(device: &CpuDevice) -> RoPE<CpuRuntime> {
        RoPE::<CpuRuntime>::precompute_freqs(128, HEAD_DIM, 10000.0, None, device)
            .expect("rope cache build must succeed")
    }

    /// A fresh single-layer paged cache with one block per row, so each
    /// row's freshly written token is the only key it can attend.
    fn test_cache(batch: usize, device: &CpuDevice) -> LayeredPagedKvCache<CpuRuntime> {
        LayeredPagedKvCache::<CpuRuntime>::new(
            1,
            batch,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            DType::F32,
            device,
        )
        .expect("paged cache build must succeed")
    }

    /// Row `b` writes into physical block `b`, slot `b * BLOCK_SIZE`.
    fn slot_mapping_and_block_table(
        batch: usize,
        device: &CpuDevice,
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let slots: Vec<i32> = (0..batch).map(|b| (b * BLOCK_SIZE) as i32).collect();
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&slots, &[batch], device).unwrap();
        let bt: Vec<i32> = (0..batch as i32).collect();
        let block_table = Tensor::<CpuRuntime>::from_slice(&bt, &[batch, 1], device).unwrap();
        (slot_mapping, block_table)
    }

    /// Deterministic, distinct-per-row decode input `[batch, 1, HIDDEN]`.
    fn decode_input(batch: usize, device: &CpuDevice) -> Var<CpuRuntime> {
        let data: Vec<f32> = (0..batch * HIDDEN)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[batch, 1, HIDDEN], device).unwrap(),
            false,
        )
    }

    /// Runs one paged decode step and returns the flat `f32` output.
    #[allow(clippy::too_many_arguments)]
    fn run_decode(
        attn: &LlamaAttention<CpuRuntime>,
        rope: &RoPE<CpuRuntime>,
        x: &Var<CpuRuntime>,
        batch: usize,
        position: usize,
        rope_positions: Option<&Tensor<CpuRuntime>>,
        device: &CpuDevice,
    ) -> Vec<f32> {
        let (client, _) = crate::test_utils::cpu_setup();
        let cache = test_cache(batch, device);
        let (slot_mapping, block_table) = slot_mapping_and_block_table(batch, device);

        let out = attn
            .forward_with_paged_kv_cache(
                &client,
                x,
                rope,
                &cache,
                0,
                &slot_mapping,
                &block_table,
                1,
                position,
                None,
                rope_positions,
            )
            .expect("paged decode forward must succeed");
        out.tensor().contiguous().unwrap().to_vec::<f32>()
    }

    /// When every row of the batch shares one RoPE position, the per-row
    /// (`RoPEPackedOps`) path must produce the exact output the scalar
    /// (`var_narrow` + `apply_rope`) path produces for that same position —
    /// the two paths agree everywhere they overlap.
    #[test]
    fn packed_positions_match_scalar_path_when_shared() {
        let (_, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let batch = 2;
        let position = 5usize;
        let x = decode_input(batch, &device);

        let scalar_out = run_decode(&attn, &rope, &x, batch, position, None, &device);

        let positions = vec![position as i32; batch];
        let positions_t = Tensor::<CpuRuntime>::from_slice(&positions, &[batch], &device).unwrap();
        let packed_out = run_decode(
            &attn,
            &rope,
            &x,
            batch,
            position, // ignored on the packed path, kept for the shared signature
            Some(&positions_t),
            &device,
        );

        assert_eq!(scalar_out.len(), packed_out.len());
        for (i, (&a, &b)) in scalar_out.iter().zip(packed_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "shared-position mismatch at {i}: scalar={a}, packed={b}"
            );
        }
    }

    /// A batch of rows at different RoPE positions must, row by row, equal
    /// that same row run alone (batch of one) at its own position — proving
    /// a mixed-position batch does not leak one row's angle into another's.
    #[test]
    fn packed_mixed_positions_match_per_row_singleton() {
        let (_, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let batch = 2;
        let positions = [3usize, 9usize];
        let x = decode_input(batch, &device);

        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let positions_t =
            Tensor::<CpuRuntime>::from_slice(&positions_i32, &[batch], &device).unwrap();
        let batch_out = run_decode(&attn, &rope, &x, batch, 0, Some(&positions_t), &device);
        let out_dim = batch_out.len() / batch;

        for (row, &pos) in positions.iter().enumerate() {
            // The same row's input, alone in a batch of one, at its own position.
            let row_data: Vec<f32> = (0..HIDDEN)
                .map(|d| ((row * HIDDEN + d) as f32 * 0.37).sin())
                .collect();
            let row_x = Var::new(
                Tensor::<CpuRuntime>::from_slice(&row_data, &[1, 1, HIDDEN], &device).unwrap(),
                false,
            );
            let row_positions_t =
                Tensor::<CpuRuntime>::from_slice(&[pos as i32], &[1], &device).unwrap();
            let singleton_out =
                run_decode(&attn, &rope, &row_x, 1, 0, Some(&row_positions_t), &device);

            let batch_row = &batch_out[row * out_dim..(row + 1) * out_dim];
            assert_eq!(singleton_out.len(), batch_row.len());
            for (i, (&a, &b)) in singleton_out.iter().zip(batch_row.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "row {row} dim {i} mismatch: singleton={a}, batched={b}"
                );
            }
        }
    }

    /// A per-row positions tensor with more than one query token per row
    /// cannot be honored (`positions` carries one id per row, not per
    /// token) and must be refused, not silently misapplied.
    #[test]
    fn packed_positions_reject_multi_token_rows() {
        let (client, device) = crate::test_utils::cpu_setup();
        let attn = test_attention(&device);
        let rope = test_rope(&device);
        let batch = 1;
        let seq_len = 2;
        let cache = test_cache(batch, &device);
        let (slot_mapping, block_table) = slot_mapping_and_block_table(batch, &device);

        let data: Vec<f32> = (0..batch * seq_len * HIDDEN).map(|i| i as f32).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[batch, seq_len, HIDDEN], &device).unwrap(),
            false,
        );
        let positions_t = Tensor::<CpuRuntime>::from_slice(&[0i32], &[batch], &device).unwrap();

        let result = attn.forward_with_paged_kv_cache(
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
            Some(&positions_t),
        );
        assert!(
            result.is_err(),
            "a per-row rope_positions tensor with seq_len > 1 must be refused"
        );
    }
}

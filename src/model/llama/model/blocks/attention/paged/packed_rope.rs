//! Per-row RoPE for a paged decode batch, via `RoPEPackedOps`.

use super::super::super::helpers::var_contiguous;
use super::super::LlamaAttention;
use crate::error::{Error, Result};
use crate::nn::RoPE;
use crate::ops::traits::RoPEPackedOps;
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Batch geometry for a packed-rope call: `batch * seq_len` gives the
/// packed row count `apply_rope_packed` expects.
#[derive(Clone, Copy)]
pub(super) struct PackedRopeShape {
    pub(super) batch: usize,
    pub(super) seq_len: usize,
}

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
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
    pub(super) fn apply_rotary_packed_paged<C>(
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
    use super::super::test_support::{
        HIDDEN, decode_input, run_decode, slot_mapping_and_block_table, test_attention, test_cache,
        test_rope,
    };
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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

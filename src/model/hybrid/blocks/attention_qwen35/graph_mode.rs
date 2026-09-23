//! CUDA graph-mode decode step for [`Qwen35AttentionBlock`]: the single-token
//! path of [`Qwen35AttentionBlock::forward`] over a full-capacity
//! [`KvCache`], with the insert position, each lane's key range and the
//! IMROPE positions read from device memory.

#[cfg(feature = "cuda")]
use super::layer::Qwen35AttentionBlock;
#[cfg(feature = "cuda")]
use super::projections::Projections;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::KvCache;
#[cfg(feature = "cuda")]
use crate::inference::decode_graph::DeviceScalars;
#[cfg(feature = "cuda")]
use crate::inference::lanes::LaneKvStep;
#[cfg(feature = "cuda")]
use crate::nn::RoPE;
#[cfg(feature = "cuda")]
use crate::nn::var_ops::var_contiguous;
#[cfg(feature = "cuda")]
use numr::autograd::{Var, var_permute, var_reshape, var_sigmoid_mul};
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

/// The per-replay state a captured `qwen35` decode step reads: each lane's
/// IMROPE positions, the device-side lengths, and each lane's key range.
/// Every buffer keeps its address across replays, which is what lets a
/// captured graph hold them.
#[cfg(feature = "cuda")]
pub struct Qwen35GraphStep<'a> {
    /// `[4, batch, 1]` i32 IMROPE streams with a stable address
    /// (`LaneTable::positions`); the gather runs on the device.
    pub positions: &'a Tensor<numr::runtime::cuda::CudaRuntime>,
    /// Device-side `seq_len_k` and `write_pos`, one entry per lane.
    pub device_scalars: &'a DeviceScalars,
    /// Each lane's key range, as `LaneTable::step` hands it out; both
    /// bounds keep their addresses across replays.
    pub lanes: LaneKvStep<'a, numr::runtime::cuda::CudaRuntime>,
}

#[cfg(feature = "cuda")]
impl Qwen35AttentionBlock<numr::runtime::cuda::CudaRuntime> {
    /// One decode step for graph capture and replay.
    ///
    /// Steps 1-3 and 5 of [`forward`](Self::forward) are the same code.
    /// Step 4 mirrors `LlamaAttention::forward_graph_mode`: `kv_insert`
    /// writes lane `b`'s k/v at `device_scalars.write_pos_ptr()[b]` and
    /// `decode_attention_graph_fwd` attends over `[kv_start[b], kv_end[b])`
    /// of the full-capacity cache, so lanes of different lengths share one
    /// captured launch.
    ///
    /// - `step.positions`, `step.device_scalars`, `step.lanes`: see
    ///   [`Qwen35GraphStep`].
    /// - `kv_cache`: pre-allocated at full capacity; `k_cache_raw()` and
    ///   `v_cache_raw()` are the addresses the graph holds. The CPU-side
    ///   `seq_len` is not advanced here.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, 1, hidden_size]` or the
    /// rope table width is not `rope_dim / 2`; kernel errors propagate.
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        x: &Var<numr::runtime::cuda::CudaRuntime>,
        rope: &RoPE<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &KvCache<numr::runtime::cuda::CudaRuntime>,
        step: &Qwen35GraphStep<'_>,
    ) -> Result<Var<numr::runtime::cuda::CudaRuntime>> {
        use crate::inference::decode_graph::insert_and_decode_attention;
        use crate::ops::traits::position::MRopeOps;

        let cfg = &self.cfg;
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != 1 || shape[2] != cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention graph mode: expected [batch, 1, {}], got {shape:?}",
                    cfg.hidden_size
                ),
            });
        }
        let batch = shape[0];
        let seq = 1usize;
        let (h, h_kv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let half_rot = cfg.rope_dim / 2;
        if rope.cos_cache().shape().get(1) != Some(&half_rot) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention graph mode: rope table must be [max_pos, {half_rot}], got {:?}",
                    rope.cos_cache().shape()
                ),
            });
        }

        // 1-2. Projections, query/gate split, per-head q/k RMS norm.
        let Projections { q, gate, k, v } = self.project(client, x, batch, seq)?;

        // 3. IMROPE with device-resident positions.
        let (cos, sin) = (rope.cos_cache(), rope.sin_cache());
        let n_rot = cfg.rope_dim;
        let q = client.mrope_interleaved_fused(
            &q,
            cos,
            sin,
            step.positions,
            &self.mrope_selector,
            n_rot,
        )?;
        let k = client.mrope_interleaved_fused(
            &k,
            cos,
            sin,
            step.positions,
            &self.mrope_selector,
            n_rot,
        )?;

        // 4. [B, 1, H, D] -> [B, H, 1, D]; insert at write_pos; attend over seq_len_k.
        let q = var_contiguous(&var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let k = var_contiguous(&var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let v = var_contiguous(&var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;

        // No sliding window: `qwen35` full-attention layers attend over the
        // whole context.
        let attn = insert_and_decode_attention(
            client,
            q.tensor(),
            k.tensor(),
            v.tensor(),
            kv_cache,
            step.device_scalars,
            Some(&step.lanes),
            h,
            h_kv,
            hd,
            0,
        )?;

        // 5. [B, H, 1, D] -> [B, 1, H, D]; gate; flatten heads; project.
        let attn = Var::new(attn, false);
        let attn = var_contiguous(&var_permute(&attn, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let gated = var_sigmoid_mul(&gate, &attn, client).map_err(Error::Numr)?;
        let gated = var_reshape(&gated, &[batch, seq, h * hd]).map_err(Error::Numr)?;
        let out = self.attn_output.forward(client, &gated)?;
        Ok(Var::new(out.tensor().clone(), false))
    }
}

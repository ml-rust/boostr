//! CUDA graph-mode attention forward — all CUDA ops, stable addresses.

#[cfg(feature = "cuda")]
use super::super::helpers::var_contiguous;
#[cfg(feature = "cuda")]
use super::LlamaAttention;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::KvCache;
#[cfg(feature = "cuda")]
use crate::nn::MaybeQuantLinear;
#[cfg(feature = "cuda")]
use numr::autograd::{Var, var_reshape};

#[cfg(feature = "cuda")]
impl LlamaAttention<numr::runtime::cuda::CudaRuntime> {
    /// Graph-mode attention forward — all CUDA ops, stable addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        x: &Var<numr::runtime::cuda::CudaRuntime>,
        cos_slice: &Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &Var<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &KvCache<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
    ) -> Result<Var<numr::runtime::cuda::CudaRuntime>>
    where
        numr::runtime::cuda::CudaClient: crate::model::traits::ModelClient<numr::runtime::cuda::CudaRuntime>
            + numr::ops::TensorOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ScalarOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ReduceOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::IndexingOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ShapeOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ActivationOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::BinaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::UnaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::CompareOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ConditionalOps<numr::runtime::cuda::CudaRuntime>,
    {
        use crate::ops::cuda::attention::flash::impl_ops::decode_attention_graph_fwd;
        use crate::ops::cuda::attention::kv_insert::kv_insert;

        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = 1usize; // graph mode is always single-token decode

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

        // Apply RoPE or skip for ALiBi models
        let (q, k) = self.apply_rotary_if_needed(client, q, k, cos_slice, sin_slice)?;

        // Insert K/V into the full-capacity cache at the device-side write_pos
        kv_insert(
            client,
            k.tensor(),
            v.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            device_scalars.write_pos_ptr(),
        )?;

        // Decode attention against the full-capacity cache with device-side seq_len_k.
        // `sliding_window` is a static config value, safe to bake into the captured
        // graph; `seq_len_k` changes per replay and stays a device pointer.
        let kv_capacity = kv_cache.capacity();
        let (attn_out, _lse) = decode_attention_graph_fwd(
            client,
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            device_scalars.seq_len_k_ptr(),
            kv_capacity,
            self.sliding_window,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = Var::new(attn_out, false);
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}

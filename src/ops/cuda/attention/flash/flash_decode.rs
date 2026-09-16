//! Flash Attention decode path: lightweight vec kernels for S_q=1.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::decode_split::{decode_dtype_suffix, decode_split_count};
use super::flash_utils::AttentionParams;

/// Kernel name stem for a supported decode `(head_dim, dtype)`.
fn decode_kernel_stem(head_dim: usize, dtype: DType) -> Result<String> {
    if head_dim != 64 && head_dim != 128 {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!("decode attention supports head_dim 64/128, got {head_dim}"),
        });
    }
    Ok(format!(
        "decode_attention_{head_dim}_{}",
        decode_dtype_suffix(dtype)?
    ))
}

/// Decode attention for S_q=1: lightweight vec kernel, no tiling.
///
/// The grid is one block per `(batch, head)` pair, which does not grow with
/// `seq_len_k`. When that leaves the device underfilled, the KV sequence is cut
/// into slices and a combine pass merges their partial softmax statistics —
/// see [`decode_split_count`].
///
/// Non-graph path: seq_len_k passed as plain i32 kernel arg (zero overhead).
pub(super) fn decode_attention_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();
    let stem = decode_kernel_stem(p.head_dim, q.dtype())?;

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;

    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, 1, p.head_dim],
        q.dtype(),
        device,
    )?;
    let lse = Tensor::<CudaRuntime>::empty(&[p.batch_size, p.num_heads, 1], DType::F32, device)?;

    let base_blocks = p.batch_size * p.num_heads;
    let splits = decode_split_count(device_index, base_blocks, p.seq_len_k);

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sk_i32 = p.seq_len_k as i32;
    let stride_i32 = kv_seq_stride as i32;
    let scale = (p.head_dim as f32).sqrt().recip();

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, p.head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&split_func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&sk_i32);
            builder.arg(&stride_i32);
            builder.arg(&scale);
            builder.arg(&splits_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("decode_attention split kernel launch failed: {:?}", e),
            })?;
        }

        let combine_func = kernels::get_kernel_function(&module, &format!("{stem}_combine"))?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&combine_func);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&o_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&splits_i32);
            builder
                .launch(combine_cfg)
                .map_err(|e| Error::KernelError {
                    reason: format!("decode_attention combine kernel launch failed: {:?}", e),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, &stem)?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
        block_dim: (p.head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&sk_i32);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}

/// Non-causal attention over a short query sequence, run as decode with the
/// query axis folded into the head axis.
///
/// `[B, H, S_q, D]` contiguous is the same bytes as `[B, H * S_q, 1, D]`, so
/// the fold is a reshape, not a copy. The decode kernel maps folded head
/// `h * S_q + s` to KV head `(h * S_q + s) / ((H * S_q) / H_kv)`; `H_kv`
/// divides `H`, so that quotient is `h / (H / H_kv)`, the same KV head the
/// unfolded row used. Output and LSE reshape back the same way.
///
/// Why: the tiled kernels give one thread per query row and stage K/V per
/// block of `block_m` rows, so a short sequence leaves most of each block
/// idle. The decode kernel spends one block per row and one warp per key.
/// `examples/cuda_short_query_profile.rs` is the measurement; the caller
/// owns the bound on `S_q`.
///
/// Non-causal only: decode has no query position to mask against. The
/// caller also excludes a sliding window.
pub(super) fn decode_attention_fwd_folded(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let folded_heads = p.num_heads * p.seq_len_q;
    let q_folded = q.reshape(&[p.batch_size, folded_heads, 1, p.head_dim])?;
    let folded = AttentionParams {
        batch_size: p.batch_size,
        num_heads: folded_heads,
        num_kv_heads: p.num_kv_heads,
        seq_len_q: 1,
        seq_len_k: p.seq_len_k,
        head_dim: p.head_dim,
        block_m: p.block_m,
        block_n: p.block_n,
        use_sm_kernel: p.use_sm_kernel,
    };
    let (output, lse) = decode_attention_fwd(client, &q_folded, k, v, &folded, kv_seq_stride)?;
    let output = output.reshape(&[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim])?;
    let lse = lse.reshape(&[p.batch_size, p.num_heads, p.seq_len_q])?;
    Ok((output, lse))
}

/// Graph-mode decode attention: uses `_graph` kernel variants with device-pointer
/// seq_len_k and separate kv_seq_stride for full-capacity raw KV buffers.
///
/// `window_size` is the sliding-window span; `0` disables it, matching every
/// other call path. Decode is single-token, so the query sits at absolute
/// position `seq_len_k - 1` and the kernel keeps keys `j >= seq_len_k -
/// window_size`. It is a static config value, not a per-step one, so passing it
/// as a plain scalar is safe under CUDA graph capture — unlike `seq_len_k`,
/// which changes every replay and therefore stays a device pointer.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_graph_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k_cache: &Tensor<CudaRuntime>,
    v_cache: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_k_ptr: u64,
    kv_capacity: usize,
    window_size: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();
    let batch_size = q.shape()[0];

    // Unlike the non-graph path, nothing upstream of graph mode filters head_dim,
    // so an unsupported one is an error, not an unreachable case.
    let stem = decode_kernel_stem(head_dim, q.dtype())?;

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1, head_dim], q.dtype(), device)?;
    let lse = Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, 1], DType::F32, device)?;

    let q_ptr = q.ptr();
    let k_ptr = k_cache.ptr();
    let v_ptr = v_cache.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = num_heads as i32;
    let nkv_i32 = num_kv_heads as i32;
    let stride_i32 = kv_capacity as i32;
    let window_i32 = window_size as i32;
    let scale = (head_dim as f32).sqrt().recip();

    let base_blocks = batch_size * num_heads;

    // The grid is baked in at capture time, so the split count cannot come from
    // the device-resident seq_len_k (only known per-replay, not at capture) — it
    // comes from kv_capacity, the static upper bound. At replay, slices whose
    // `[begin, end)` falls past the real seq_len_k are empty; the split kernel's
    // `begin < end` guard and the combine kernel's `l <= 0` guard both skip them
    // for free. Consequence: at early decode steps, with the cache nearly empty,
    // most slices do no work — correct, but the grid stays sized for a full cache
    // every step, not just the steps that need it.
    let splits = decode_split_count(device_index, base_blocks, kv_capacity);

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics.
        // Allocated inside the capture closure: numr's allocator is frozen during
        // capture, so this becomes a graph alloc/free node pair replayed every
        // launch, exactly like the non-graph split path's scratch.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split_graph"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&split_func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&seq_len_k_ptr);
            builder.arg(&stride_i32);
            builder.arg(&scale);
            builder.arg(&window_i32);
            builder.arg(&splits_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("decode_attention_graph split kernel launch failed: {:?}", e),
            })?;
        }

        // Static num_splits, so the combine kernel is capture-safe unchanged —
        // the same entry point the non-graph split path already uses.
        let combine_func = kernels::get_kernel_function(&module, &format!("{stem}_combine"))?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&combine_func);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&o_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&splits_i32);
            builder
                .launch(combine_cfg)
                .map_err(|e| Error::KernelError {
                    reason: format!(
                        "decode_attention_graph combine kernel launch failed: {:?}",
                        e
                    ),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, &format!("{stem}_graph"))?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&seq_len_k_ptr);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.arg(&window_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention_graph kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::attention::{StandardAttnConfig, standard_attention_fwd};
    use crate::ops::traits::FlashAttentionOps;
    use crate::test_utils::cuda_setup;
    use numr::runtime::cuda::CudaDevice;

    /// Deterministic values with no repeated rows, so a wrong KV-head mapping
    /// or a mis-strided query row changes the answer.
    fn filled(shape: &[usize], seed: f32, device: &CudaDevice) -> Tensor<CudaRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| 0.5 * ((i as f32) * 0.731 + seed).sin())
            .collect();
        Tensor::<CudaRuntime>::from_slice(&data, shape, device).expect("tensor")
    }

    fn max_abs_diff(a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> f32 {
        let a = a.contiguous().expect("contiguous").to_vec::<f32>();
        let b = b.contiguous().expect("contiguous").to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Every geometry the fold admits: GQA, several query rows per head, and
    /// a KV sequence longer than the query one. The reference is the
    /// composed Tensor path, which expands KV heads explicitly.
    #[test]
    fn folded_decode_matches_standard_attention() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        for &(heads, kv_heads, head_dim, seq_q, seq_k) in &[
            (16usize, 2usize, 128usize, 11usize, 11usize),
            (16, 2, 128, 5, 5),
            (4, 4, 64, 7, 7),
            (8, 1, 64, 3, 40),
        ] {
            let q = filled(&[2, heads, seq_q, head_dim], 0.1, device);
            let k = filled(&[2, kv_heads, seq_k, head_dim], 0.2, device);
            let v = filled(&[2, kv_heads, seq_k, head_dim], 0.3, device);

            let (out, lse) = client
                .flash_attention_fwd(&q, &k, &v, heads, kv_heads, head_dim, false, 0, None)
                .expect("flash_attention_fwd");
            let cfg = StandardAttnConfig {
                num_heads: heads,
                num_kv_heads: kv_heads,
                causal: false,
                window_size: 0,
            };
            let (ref_out, ref_lse) =
                standard_attention_fwd(client, &q, &k, &v, cfg).expect("standard_attention_fwd");

            assert_eq!(out.shape(), &[2, heads, seq_q, head_dim]);
            assert_eq!(lse.shape(), &[2, heads, seq_q]);
            let out_diff = max_abs_diff(&out, &ref_out);
            let lse_diff = max_abs_diff(&lse, &ref_lse);
            assert!(
                out_diff < 1e-4,
                "output diverges at H={heads} Hkv={kv_heads} D={head_dim} Sq={seq_q} Sk={seq_k}: {out_diff}"
            );
            assert!(
                lse_diff < 1e-4,
                "lse diverges at H={heads} Hkv={kv_heads} D={head_dim} Sq={seq_q} Sk={seq_k}: {lse_diff}"
            );
        }
    }
}

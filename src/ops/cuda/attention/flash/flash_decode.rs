//! Flash Attention decode path: lightweight vec kernels for S_q=1.
//!
//! The graph-capturable variant lives in `flash_decode_graph.rs`.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use crate::ops::traits::AttnOutLayout;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::decode_split::{
    DECODE_HEAD_DIMS, decode_dtype_suffix, decode_kv_span, decode_split_count,
    decode_supports_head_dim,
};
use super::flash_utils::AttentionParams;

/// Kernel name stem for a supported decode `(head_dim, dtype)`.
pub(super) fn decode_kernel_stem(head_dim: usize, dtype: DType) -> Result<String> {
    if !decode_supports_head_dim(head_dim) {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "decode attention supports head_dim {DECODE_HEAD_DIMS:?}, got {head_dim}"
            ),
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
/// `window_size` follows the flash contract (`0` = unlimited). The single
/// query sits at position `seq_len_k - 1`, so a window keeps the last
/// `min(window_size, seq_len_k)` keys; the kernel starts its key loop there
/// and the split count is sized to that span, not the full sequence. The
/// causal flag has no effect at `seq_len_q == 1` and is not passed.
///
/// `out_layout`: at one query row `[B, H, 1, D]` and `[B, 1, H, D]` are the
/// same bytes, so the kernel stores head-major and the result is reshaped.
///
/// `kv_start`: device pointer to the `[B]` I32 left-padding starts, or null.
/// The kernel raises each row's key-loop start to it, after the window.
///
/// Non-graph path: seq_len_k passed as plain i32 kernel arg (zero overhead).
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
    window_size: usize,
    kv_start: u64,
    out_layout: AttnOutLayout,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    if p.seq_len_q != 1 {
        return Err(Error::InvalidArgument {
            arg: "seq_len_q",
            reason: format!("decode attention serves one query row, got {}", p.seq_len_q),
        });
    }
    let (output, lse) = launch_decode(client, q, k, v, p, kv_seq_stride, window_size, kv_start, 1)?;
    let output = output.reshape(&out_layout.shape(p.batch_size, p.num_heads, 1, p.head_dim))?;
    Ok((output, lse))
}

/// Launches the decode kernels for `p.num_heads` rows per batch and returns
/// the output as `[B, num_heads, 1, D]` plus the LSE `[B, num_heads, 1]`.
///
/// `out_seq_fold` is the kernel's output-row rule: `1` stores row `bh` at
/// `bh * D`; `S_q > 1` means `num_heads` is a fold `H * S_q` and the kernel
/// stores row `h * S_q + s` at token-major `o[b, s, h, :]`. The returned
/// tensor's shape describes the bytes only in the `1` case; the folded
/// caller reshapes.
#[allow(clippy::too_many_arguments)]
fn launch_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
    window_size: usize,
    kv_start: u64,
    out_seq_fold: usize,
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
    let splits = decode_split_count(
        device_index,
        base_blocks,
        decode_kv_span(p.seq_len_k, window_size),
        p.head_dim,
    );

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sk_i32 = p.seq_len_k as i32;
    let stride_i32 = kv_seq_stride as i32;
    let window_i32 = window_size as i32;
    let fold_i32 = out_seq_fold as i32;
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
            builder.arg(&window_i32);
            builder.arg(&kv_start);
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
            builder.arg(&nh_i32);
            builder.arg(&fold_i32);
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
        builder.arg(&window_i32);
        builder.arg(&kv_start);
        builder.arg(&fold_i32);
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
///
/// Token-major output: the kernel unfolds row `h * S_q + s` back to
/// `o[b, s, h, :]` itself (`out_seq_fold = S_q`), so no copy follows.
///
/// `kv_start` (device pointer or null) is indexed by the batch row, which
/// the kernel recovers as `bh / (H * S_q)`; the fold leaves it untouched.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_fwd_folded(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
    kv_start: u64,
    out_layout: AttnOutLayout,
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
    let out_seq_fold = match out_layout {
        AttnOutLayout::HeadMajor => 1,
        AttnOutLayout::TokenMajor => p.seq_len_q,
    };
    let (output, lse) = launch_decode(
        client,
        &q_folded,
        k,
        v,
        &folded,
        kv_seq_stride,
        0,
        kv_start,
        out_seq_fold,
    )?;
    let output =
        output.reshape(&out_layout.shape(p.batch_size, p.num_heads, p.seq_len_q, p.head_dim))?;
    let lse = lse.reshape(&[p.batch_size, p.num_heads, p.seq_len_q])?;
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

    /// Single-query decode at every head_dim the kernel instantiates, with and
    /// without a window, on both grid shapes, against the composed Tensor
    /// path. `causal` is run both ways: at `S_q == 1` the query is the newest
    /// position, so the flag must not change the answer.
    #[test]
    fn decode_matches_standard_attention() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        for &head_dim in &[32usize, 64, 96, 128, 192, 256] {
            for &(seq_k, window) in &[(40usize, 0usize), (40, 12), (700, 0), (700, 450)] {
                for &causal in &[false, true] {
                    let q = filled(&[2, 8, 1, head_dim], 0.1, device);
                    let k = filled(&[2, 2, seq_k, head_dim], 0.2, device);
                    let v = filled(&[2, 2, seq_k, head_dim], 0.3, device);

                    let (out, lse) = client
                        .flash_attention_fwd(
                            &q,
                            &k,
                            &v,
                            8,
                            2,
                            head_dim,
                            causal,
                            window,
                            None,
                            None,
                            AttnOutLayout::HeadMajor,
                        )
                        .expect("flash_attention_fwd");
                    let cfg = StandardAttnConfig {
                        num_heads: 8,
                        num_kv_heads: 2,
                        causal,
                        window_size: window,
                    };
                    let (ref_out, ref_lse) = standard_attention_fwd(client, &q, &k, &v, cfg)
                        .expect("standard_attention_fwd");

                    let out_diff = max_abs_diff(&out, &ref_out);
                    let lse_diff = max_abs_diff(&lse, &ref_lse);
                    assert!(
                        out_diff < 1e-4,
                        "output diverges at D={head_dim} Sk={seq_k} window={window} causal={causal}: {out_diff}"
                    );
                    assert!(
                        lse_diff < 1e-4,
                        "lse diverges at D={head_dim} Sk={seq_k} window={window} causal={causal}: {lse_diff}"
                    );
                }
            }
        }
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
                .flash_attention_fwd(
                    &q,
                    &k,
                    &v,
                    heads,
                    kv_heads,
                    head_dim,
                    false,
                    0,
                    None,
                    None,
                    AttnOutLayout::HeadMajor,
                )
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

//! Flash Attention decode path: lightweight vec kernels for S_q=1.
//!
//! Kernel launch plumbing (module load, grid sizing, launch builders) lives
//! in `flash_decode_launch.rs`; this file holds the two entry points that
//! shape their arguments and reshape the result.
//!
//! The graph-capturable variant lives in `flash_decode_graph.rs`.

use crate::error::{Error, Result};
use crate::ops::traits::AttnOutLayout;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

pub(super) use super::flash_decode_launch::decode_kernel_stem;
use super::flash_decode_launch::launch_decode;
use super::flash_utils::AttentionParams;

/// Decode attention for S_q=1: lightweight vec kernel, no tiling.
///
/// The grid is one block per `(batch, head)` pair, which does not grow with
/// `seq_len_k`. When the span is longer than one slice, the KV sequence is
/// cut into slices at fixed absolute positions and a combine pass merges
/// their partial softmax statistics in slice order — see `decode_slices` in
/// `ops/cuda/attention/decode_split.rs`. The cut never reads the batch size,
/// so each row's output is the same bits as its own single-row decode.
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
/// `kv_end`: device pointer to the `[B]` I32 per-row key ends, or null. Row
/// `b` then attends over `[kv_start[b], min(kv_end[b], p.seq_len_k))`. Null
/// leaves every row ending at `p.seq_len_k`.
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
    kv_end: u64,
    out_layout: AttnOutLayout,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    if p.seq_len_q != 1 {
        return Err(Error::InvalidArgument {
            arg: "seq_len_q",
            reason: format!("decode attention serves one query row, got {}", p.seq_len_q),
        });
    }
    let (output, lse) = launch_decode(
        client,
        q,
        k,
        v,
        p,
        kv_seq_stride,
        window_size,
        kv_start,
        kv_end,
        1,
    )?;
    let output = output.reshape(&out_layout.shape(p.batch_size, p.num_heads, 1, p.head_dim))?;
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
        0,
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

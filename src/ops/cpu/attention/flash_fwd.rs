//! CPU `flash_attention_fwd`: the fused F32 decode loop for a single query
//! row, and the composed standard attention for everything else.
//!
//! `kv_start` and a per-row `KvSeqLen::PerRow` end are read to the host once
//! per call and applied by both paths: the decode loop walks each batch row
//! over `[start_b, end_b)`, and the standard path runs each such row alone
//! over its own keys (`rows_from_starts`), so the row's result is the bits
//! its unpadded run forms. One corner keeps the per-row additive mask
//! (`standard_attention_fwd_kv_start`): a window without a causal mask on a
//! row whose valid keys are fewer than its queries, where narrowing the keys
//! would move the window's anchor.

use crate::error::Result;
use crate::ops::impl_generic::attention::{
    StandardAttnConfig, clamped_kv_ends, standard_attention_fwd, standard_attention_fwd_kv_start,
    validate_kv_start, validate_row_bounds,
};
use crate::ops::traits::{AttnOutLayout, KvSeqLen};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Body of `FlashAttentionOps::flash_attention_fwd` for `CpuClient`.
#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_cpu(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window_size: usize,
    kv_seq_len: Option<KvSeqLen<'_, CpuRuntime>>,
    kv_start: Option<&Tensor<CpuRuntime>>,
    out_layout: AttnOutLayout,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    // A shared length narrows K/V once for the whole batch, then runs as if
    // the buffer were that long.
    if let Some(KvSeqLen::Shared(seq_len)) = kv_seq_len {
        let k_narrow = k.narrow(2, 0, seq_len)?;
        let v_narrow = v.narrow(2, 0, seq_len)?;
        let k_c = k_narrow.contiguous()?;
        let v_c = v_narrow.contiguous()?;
        return flash_attention_fwd_cpu(
            client,
            q,
            &k_c,
            &v_c,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
            None,
            kv_start,
            out_layout,
        );
    }

    let batch = q.shape()[0];
    // The start vector is read to the host once; both paths below index
    // it per batch row.
    let starts: Option<Vec<i32>> = match kv_start {
        Some(t) => {
            validate_kv_start(t, batch)?;
            Some(t.to_vec::<i32>())
        }
        None => None,
    };

    // Per-row ends are read the same way and checked against each row's
    // start and the key axis before any kernel runs.
    let ends: Option<Vec<usize>> = match kv_seq_len {
        Some(KvSeqLen::PerRow(t)) => {
            validate_row_bounds(t, batch, "kv_seq_len")?;
            Some(clamped_kv_ends(
                &t.to_vec::<i32>(),
                starts.as_deref(),
                k.shape()[2],
            )?)
        }
        _ => None,
    };

    // Fast path: fused decode attention for S_q=1 (single token generation)
    // Avoids all intermediate tensor allocations and GQA expansion
    let seq_len_q = q.shape()[2];
    if seq_len_q == 1
        && !causal
        && window_size == 0
        && q.dtype() == DType::F32
        && k.dtype() == DType::F32
        && v.dtype() == DType::F32
    {
        // `[B, H, 1, D]` and `[B, 1, H, D]` are the same bytes.
        let (out, lse) = super::decode_attention::fused_decode_attention(
            q,
            k,
            v,
            num_heads,
            num_kv_heads,
            head_dim,
            super::decode_attention::RowRange {
                kv_start: starts.as_deref(),
                kv_end: ends.as_deref(),
            },
        )?;
        return Ok((out_layout.from_head_major(out)?, lse));
    }

    let _ = head_dim; // validated by shape
    let cfg = StandardAttnConfig {
        num_heads,
        num_kv_heads,
        causal,
        window_size,
    };
    // The composed path ends in a batched matmul, whose output is
    // head-major by construction; token-major is that result re-laid,
    // element for element.
    let per_row = ends.is_some() || starts.as_deref().is_some_and(|s| s.iter().any(|&x| x > 0));
    let (out, lse) = if per_row {
        rows_from_starts(client, q, k, v, cfg, starts.as_deref(), ends.as_deref())?
    } else {
        standard_attention_fwd(client, q, k, v, cfg)?
    };
    Ok((out_layout.from_head_major(out)?, lse))
}

/// Query rows of a batch row that see no key: with a causal mask, the rows
/// whose absolute position `S_k - S_q + i` is below `start`; without it,
/// every row once `start` reaches `S_k`. The same rule the mask builder
/// applies (`build_kv_start_masks`).
fn dead_rows(start: usize, seq_len_q: usize, seq_len_k: usize, causal: bool) -> usize {
    if start >= seq_len_k {
        return seq_len_q;
    }
    if !causal {
        return 0;
    }
    let key_offset = seq_len_k.saturating_sub(seq_len_q);
    start.saturating_sub(key_offset).min(seq_len_q)
}

/// One row's keys cut to its own end. A row that ends at the buffer's end
/// keeps the view it already has, so a call without per-row ends copies
/// nothing.
fn narrow_to_end(
    kv: &Tensor<CpuRuntime>,
    seq_len_k: usize,
    buffer_len: usize,
) -> Result<Tensor<CpuRuntime>> {
    if seq_len_k == buffer_len {
        return Ok(kv.clone());
    }
    Ok(kv.narrow(2, 0, seq_len_k)?.contiguous()?)
}

/// The composed attention, one batch row at a time, each over the keys
/// between its own start and its own end, and the query rows that see any
/// key. With the dead rows dropped the narrowed call's key offset places
/// every live query at the position the full call gives it, so the causal
/// and window bounds are the same and the row's floats are the ones its
/// unpadded run forms. Dead rows store zeros with an LSE of `-inf`.
///
/// A window without a causal mask on a row with fewer valid keys than
/// queries has no such offset: the key offset saturates at zero and the
/// window would anchor elsewhere. That row keeps the additive-mask path,
/// over its keys narrowed to its own end.
fn rows_from_starts(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    cfg: StandardAttnConfig,
    starts: Option<&[i32]>,
    ends: Option<&[usize]>,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let (batch, num_heads, seq_len_q, head_dim) =
        (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let buffer_len = k.shape()[2];
    let device = q.device();
    let mut outs = Vec::with_capacity(batch);
    let mut lses = Vec::with_capacity(batch);
    for b in 0..batch {
        let seq_len_k = ends.map_or(buffer_len, |e| e[b]);
        let start = starts
            .map_or(0, |s| usize::try_from(s[b].max(0)).unwrap_or(0))
            .min(seq_len_k);
        let dead = dead_rows(start, seq_len_q, seq_len_k, cfg.causal);
        let live = seq_len_q - dead;
        let q_b = q.narrow(0, b, 1)?;
        let k_b = k.narrow(0, b, 1)?;
        let v_b = v.narrow(0, b, 1)?;
        let window_anchor_moves =
            !cfg.causal && cfg.window_size > 0 && seq_len_k - start < seq_len_q;
        let (out_b, lse_b) = if live == 0 {
            (
                Tensor::<CpuRuntime>::zeros(
                    &[1, num_heads, seq_len_q, head_dim],
                    q.dtype(),
                    device,
                )?,
                Tensor::<CpuRuntime>::full_scalar(
                    &[1, num_heads, seq_len_q],
                    DType::F32,
                    f64::NEG_INFINITY,
                    device,
                )?,
            )
        } else if window_anchor_moves {
            let k_end = narrow_to_end(&k_b, seq_len_k, buffer_len)?;
            let v_end = narrow_to_end(&v_b, seq_len_k, buffer_len)?;
            let start = i32::try_from(start).unwrap_or(i32::MAX);
            standard_attention_fwd_kv_start(client, &q_b, &k_end, &v_end, cfg, Some(&[start]))?
        } else {
            let q_live = q_b.narrow(2, dead, live)?.contiguous()?;
            let k_live = k_b.narrow(2, start, seq_len_k - start)?.contiguous()?;
            let v_live = v_b.narrow(2, start, seq_len_k - start)?.contiguous()?;
            let (o, l) = standard_attention_fwd(client, &q_live, &k_live, &v_live, cfg)?;
            if dead == 0 {
                (o, l)
            } else {
                let o_dead = Tensor::<CpuRuntime>::zeros(
                    &[1, num_heads, dead, head_dim],
                    q.dtype(),
                    device,
                )?;
                let l_dead = Tensor::<CpuRuntime>::full_scalar(
                    &[1, num_heads, dead],
                    DType::F32,
                    f64::NEG_INFINITY,
                    device,
                )?;
                (
                    Tensor::cat(&[&o_dead, &o], 2)?,
                    Tensor::cat(&[&l_dead, &l], 2)?,
                )
            }
        };
        outs.push(out_b);
        lses.push(lse_b);
    }
    let out_refs: Vec<&Tensor<CpuRuntime>> = outs.iter().collect();
    let lse_refs: Vec<&Tensor<CpuRuntime>> = lses.iter().collect();
    Ok((Tensor::cat(&out_refs, 0)?, Tensor::cat(&lse_refs, 0)?))
}

#[cfg(test)]
mod tests {
    use crate::ops::traits::{AttnOutLayout, FlashAttentionOps};
    use crate::test_utils::cpu_setup;
    use numr::ops::{BinaryOps, ReduceOps, UnaryOps};
    use numr::runtime::cpu::{CpuClient, CpuRuntime};
    use numr::tensor::Tensor;

    fn rand_tensor(
        shape: &[usize],
        _client: &CpuClient,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        // Simple deterministic pseudo-random data
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    #[test]
    fn test_flash_fwd_output_shape() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (2, 4, 8, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_flash_fwd_causal() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 6, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_causal, _) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                true,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();

        // Causal and full should differ (unless trivial inputs)
        let diff = client.sub(&out_causal, &out_full).unwrap();
        let abs_diff = client.abs(&diff).unwrap();
        let max_diff = client.max(&abs_diff, &[], false).unwrap();
        let max_val = max_diff.to_vec::<f32>()[0];
        assert!(
            max_val > 1e-6,
            "Causal and non-causal outputs should differ"
        );
    }

    #[test]
    fn test_flash_fwd_sliding_window() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 12, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_window, _) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                false,
                4,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                h,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();

        let ow = out_window.to_vec::<f32>();
        let of = out_full.to_vec::<f32>();
        let max_diff = ow
            .iter()
            .zip(of.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Sliding window should differ from full attention"
        );
    }

    #[test]
    fn test_flash_fwd_gqa() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, s, d) = (1, 8, 2, 4, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, nkv, s, d], &client, &device);
        let v = rand_tensor(&[b, nkv, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                h,
                nkv,
                d,
                false,
                0,
                None,
                None,
                AttnOutLayout::HeadMajor,
            )
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }
}

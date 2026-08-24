//! Standard (dense, O(N²)) attention forward/backward on plain tensors.
//!
//! THE algorithm — same for all backends. Composed from numr ops, so any
//! backend whose client implements the listed op traits can delegate here
//! instead of writing a fused kernel. Fused kernels (e.g. CUDA FlashAttention,
//! WGSL forward) are optimizations that MUST match these results.
//!
//! Forward returns `(output, logsumexp)` matching the FlashAttention interface.
//! Backward recomputes the attention weights from `q, k` (it does not need the
//! saved `lse`) and returns `(dq, dk, dv)`, reducing GQA head groups back to
//! `num_kv_heads`.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trait bundle a client must satisfy to run standard attention.
pub trait StandardAttentionClient<R: Runtime>:
    RuntimeClient<R>
    + MatmulOps<R>
    + ScalarOps<R>
    + BinaryOps<R>
    + ActivationOps<R>
    + ReduceOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> StandardAttentionClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + MatmulOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + ActivationOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>,
{
}

/// Geometry + masking configuration shared by standard attention fwd/bwd.
#[derive(Debug, Clone, Copy)]
pub struct StandardAttnConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (`< num_heads` ⇒ GQA).
    pub num_kv_heads: usize,
    /// Apply a causal mask.
    pub causal: bool,
    /// Sliding-window width (`0` = unlimited).
    pub window_size: usize,
}

/// Standard attention forward: `softmax(Q·Kᵀ/√d + mask)·V`.
///
/// `q`/`k`/`v` are `[B, H, S, D]` (`k`/`v` use `cfg.num_kv_heads` for GQA).
/// Returns `(output [B, H, S_q, D], logsumexp [B, H, S_q] (F32))`.
pub fn standard_attention_fwd<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    cfg: StandardAttnConfig,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R>,
{
    let q_shape = q.shape();
    let head_dim = q_shape[3];
    let seq_len_q = q_shape[2];
    let seq_len_k = k.shape()[2];
    let scale = (head_dim as f64).sqrt().recip();

    let (k_expanded, v_expanded) = expand_kv(client, k, v, cfg.num_heads, cfg.num_kv_heads)?;

    // Q @ K^T → [B, H, S_q, S_k]
    let k_t = k_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let k_t = k_t.contiguous()?;
    let scores = client.matmul(q, &k_t).map_err(Error::Numr)?;
    let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;

    let scores = apply_mask(
        client,
        scores,
        seq_len_q,
        seq_len_k,
        cfg.causal,
        cfg.window_size,
        q.device(),
    )?;

    // Logsumexp for the backward pass: [B, H, S_q]
    let lse = client
        .logsumexp(&scores, &[3], false)
        .map_err(Error::Numr)?;

    let weights = client.softmax(&scores, -1).map_err(Error::Numr)?;
    let output = client.matmul(&weights, &v_expanded).map_err(Error::Numr)?;

    let lse = if lse.dtype() != DType::F32 {
        client.cast(&lse, DType::F32).map_err(Error::Numr)?
    } else {
        lse
    };

    Ok((output, lse))
}

/// Standard attention backward. Recomputes weights from `q, k`, then returns
/// `(dq [B, H, S_q, D], dk [B, num_kv_heads, S_k, D], dv [...])`.
pub fn standard_attention_bwd<R, C>(
    client: &C,
    dout: &Tensor<R>,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    output: &Tensor<R>,
    cfg: StandardAttnConfig,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R>,
{
    let q_shape = q.shape();
    let head_dim = q_shape[3];
    let seq_len_q = q_shape[2];
    let seq_len_k = k.shape()[2];
    let scale = (head_dim as f64).sqrt().recip();

    let (k_expanded, v_expanded) = expand_kv(client, k, v, cfg.num_heads, cfg.num_kv_heads)?;

    // Recompute attention weights.
    let k_t = k_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let k_t = k_t.contiguous()?;
    let scores = client.matmul(q, &k_t).map_err(Error::Numr)?;
    let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;
    let scores = apply_mask(
        client,
        scores,
        seq_len_q,
        seq_len_k,
        cfg.causal,
        cfg.window_size,
        q.device(),
    )?;
    let weights = client.softmax(&scores, -1).map_err(Error::Numr)?;

    // dV = Pᵀ @ dO  → [B, H, S_k, D]
    let weights_t = weights.transpose(-2, -1).map_err(Error::Numr)?;
    let weights_t = weights_t.contiguous()?;
    let dv_expanded = client.matmul(&weights_t, dout).map_err(Error::Numr)?;

    // dP = dO @ Vᵀ  → [B, H, S_q, S_k]
    let v_t = v_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let v_t = v_t.contiguous()?;
    let dp = client.matmul(dout, &v_t).map_err(Error::Numr)?;

    // D = rowsum(dO ⊙ O)  → [B, H, S_q, 1]
    let do_times_o = client.mul(dout, output).map_err(Error::Numr)?;
    let d_buf = client.sum(&do_times_o, &[3], false).map_err(Error::Numr)?;

    // dS = P ⊙ (dP − D)  → [B, H, S_q, S_k]
    let d_expanded = d_buf.unsqueeze(-1).map_err(Error::Numr)?;
    let d_expanded = d_expanded.broadcast_to(dp.shape()).map_err(Error::Numr)?;
    let ds = client.sub(&dp, &d_expanded).map_err(Error::Numr)?;
    let ds = client.mul(&weights, &ds).map_err(Error::Numr)?;
    let ds = client.mul_scalar(&ds, scale).map_err(Error::Numr)?;

    // dQ = dS @ K  → [B, H, S_q, D]
    let dq = client.matmul(&ds, &k_expanded).map_err(Error::Numr)?;

    // dK = dSᵀ @ Q  → [B, H, S_k, D]
    let ds_t = ds.transpose(-2, -1).map_err(Error::Numr)?;
    let ds_t = ds_t.contiguous()?;
    let dk_expanded = client.matmul(&ds_t, q).map_err(Error::Numr)?;

    // Reduce GQA head groups back to num_kv_heads.
    let (dk, dv) = if cfg.num_kv_heads < cfg.num_heads {
        let repeats = cfg.num_heads / cfg.num_kv_heads;
        let dk = sum_gqa_grads(client, &dk_expanded, cfg.num_kv_heads, repeats)?;
        let dv = sum_gqa_grads(client, &dv_expanded, cfg.num_kv_heads, repeats)?;
        (dk, dv)
    } else {
        (dk_expanded, dv_expanded)
    };

    Ok((dq, dk, dv))
}

/// GQA: expand `num_kv_heads` KV heads up to `num_heads` by repeating each head.
fn expand_kv<R, C>(
    client: &C,
    k: &Tensor<R>,
    v: &Tensor<R>,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R>,
{
    if num_kv_heads < num_heads {
        let repeats = num_heads / num_kv_heads;
        let k_exp = client
            .repeat_interleave(k, repeats, Some(1))
            .map_err(Error::Numr)?;
        let v_exp = client
            .repeat_interleave(v, repeats, Some(1))
            .map_err(Error::Numr)?;
        Ok((k_exp, v_exp))
    } else {
        Ok((k.clone(), v.clone()))
    }
}

/// Add the causal / sliding-window additive mask to `scores` if either applies.
fn apply_mask<R, C>(
    client: &C,
    scores: Tensor<R>,
    seq_len_q: usize,
    seq_len_k: usize,
    causal: bool,
    window_size: usize,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R>,
{
    if !(causal || window_size > 0) {
        return Ok(scores);
    }
    let mask = build_attention_mask::<R>(seq_len_q, seq_len_k, causal, window_size, device)?;
    let mask = if mask.dtype() != scores.dtype() {
        client.cast(&mask, scores.dtype()).map_err(Error::Numr)?
    } else {
        mask
    };
    client.add(&scores, &mask).map_err(Error::Numr)
}

/// Sum gradients from expanded heads back to `num_kv_heads` for GQA.
/// `[B, num_kv_heads·repeats, S, D]` → `[B, num_kv_heads, S, D]`.
pub fn sum_gqa_grads<R, C>(
    client: &C,
    grad: &Tensor<R>,
    num_kv_heads: usize,
    repeats: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: StandardAttentionClient<R>,
{
    let shape = grad.shape();
    let (b, _nh, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    let reshaped = grad
        .reshape(&[b, num_kv_heads, repeats, s, d])
        .map_err(Error::Numr)?;
    client.sum(&reshaped, &[2], false).map_err(Error::Numr)
}

/// Build the additive attention mask: `0` where allowed, `-inf` where masked.
///
/// Query row `i` is at ABSOLUTE sequence position `key_offset + i`, where
/// `key_offset = seq_len_k - seq_len_q`. A KV-cached decode or chunked prefill
/// passes `seq_len_q` new queries against `seq_len_k` cached keys, so the
/// queries are the LAST `seq_len_q` positions of the key sequence. Prefill and
/// training pass `seq_len_q == seq_len_k`, giving `key_offset == 0` and leaving
/// both rules exactly as they were. Same convention as
/// `model/attention_mask.rs::causal_window_mask` and `kernels/attention/flash_v2.cu`.
///
/// - causal: position `j > key_offset + i` is masked
/// - window_size > 0: position `j + window_size <= key_offset + i` is masked
///   (`0` disables the window; the window is INCLUSIVE of the current token, so
///   row `i` keeps exactly `window_size` keys)
pub fn build_attention_mask<R: Runtime<DType = DType>>(
    seq_len_q: usize,
    seq_len_k: usize,
    causal: bool,
    window_size: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let key_offset = seq_len_k.saturating_sub(seq_len_q);
    let mut mask_data = vec![0.0f32; seq_len_q * seq_len_k];
    for i in 0..seq_len_q {
        let q_pos = key_offset + i;
        for j in 0..seq_len_k {
            let masked = (causal && j > q_pos) || (window_size > 0 && (j + window_size) <= q_pos);
            if masked {
                mask_data[i * seq_len_k + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::<R>::try_from_slice(
        &mask_data,
        &[1, 1, seq_len_q, seq_len_k],
        device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// The mask rule as it was written before absolute positions: query index
    /// `i` taken as a position within the query tensor.
    fn legacy_masked(i: usize, j: usize, causal: bool, window_size: usize) -> bool {
        (causal && j > i) || (window_size > 0 && (j + window_size) <= i)
    }

    fn mask(sq: usize, sk: usize, causal: bool, window_size: usize) -> Vec<f32> {
        let (_client, device) = cpu_setup();
        build_attention_mask::<CpuRuntime>(sq, sk, causal, window_size, &device)
            .unwrap()
            .to_vec::<f32>()
    }

    /// Prefill (`sq == sk`) has `key_offset == 0`, so every entry must be
    /// bit-identical to the legacy relative-index rule — windowed and not.
    #[test]
    fn prefill_mask_is_bit_identical_to_legacy() {
        let s = 7;
        for (causal, window_size) in [(false, 3), (true, 0), (true, 3), (false, 1)] {
            let got = mask(s, s, causal, window_size);
            for i in 0..s {
                for j in 0..s {
                    let want = if legacy_masked(i, j, causal, window_size) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                    assert_eq!(
                        got[i * s + j],
                        want,
                        "prefill mask changed at ({i},{j}) causal={causal} window={window_size}"
                    );
                }
            }
        }
    }

    /// Single-token decode: the one query row sits at absolute position
    /// `sk - 1`, so a window keeps exactly the last `window_size` keys.
    #[test]
    fn decode_mask_keeps_only_the_window_suffix() {
        let (sk, window_size) = (8usize, 3usize);
        let got = mask(1, sk, false, window_size);
        for (j, &m) in got.iter().enumerate() {
            let allowed = j >= sk - window_size;
            assert_eq!(
                m == 0.0,
                allowed,
                "decode window: key {j} allowed={allowed} but mask={m}"
            );
        }
    }

    /// The causal clause is inert on decode: the query is the newest position,
    /// so no key is in its future and the row is never fully masked.
    #[test]
    fn decode_mask_causal_masks_nothing() {
        let got = mask(1, 8, true, 0);
        assert!(got.iter().all(|&m| m == 0.0), "causal decode masked keys");
    }

    /// Chunked prefill (`1 < sq < sk`): row `i` is at absolute position
    /// `sk - sq + i` and keeps `window_size` keys ending there.
    #[test]
    fn chunked_mask_uses_absolute_positions() {
        let (sq, sk, window_size) = (3usize, 8usize, 3usize);
        let got = mask(sq, sk, true, window_size);
        let key_offset = sk - sq;
        for i in 0..sq {
            let q_pos = key_offset + i;
            for j in 0..sk {
                let allowed = j <= q_pos && j + window_size > q_pos;
                assert_eq!(
                    got[i * sk + j] == 0.0,
                    allowed,
                    "chunked mask at ({i},{j}): expected allowed={allowed}"
                );
            }
        }
    }
}

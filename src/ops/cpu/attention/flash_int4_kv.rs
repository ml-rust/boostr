//! CPU reference for `FlashAttentionOps::flash_attention_fwd_int4_kv`.
//!
//! Not a fused kernel: dequantizes K/V from packed INT4 to F32 via the
//! shared `KvCacheQuantOps::dequantize_kv_int4` path, then runs the same
//! standard attention CPU `flash_attention_fwd` falls back to.

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::{StandardAttnConfig, standard_attention_fwd};
use crate::ops::traits::cache::kv_cache_quant::{Int4GroupSize, KvCacheQuantOps};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Dequantize a packed INT4 KV tensor `[batch, heads, seq_len, head_dim/2]`
/// to F32 `[batch, heads, seq_len, head_dim]`, delegating the nibble/scale/
/// zero unpack to `KvCacheQuantOps::dequantize_kv_int4` one `[seq_len,
/// head_dim]` slice at a time — never a second copy of that layout.
#[allow(clippy::too_many_arguments)]
fn dequant_int4_kv(
    client: &CpuClient,
    quant_name: &'static str,
    quant: &Tensor<CpuRuntime>,
    scales_name: &'static str,
    scales: &Tensor<CpuRuntime>,
    zeros_name: &'static str,
    zeros: &Tensor<CpuRuntime>,
    group_size: Int4GroupSize,
) -> Result<Tensor<CpuRuntime>> {
    if quant.dtype() != DType::U8 {
        return Err(Error::InvalidArgument {
            arg: quant_name,
            reason: format!(
                "flash_attention_fwd_int4_kv requires {quant_name} in U8, got {:?}",
                quant.dtype()
            ),
        });
    }
    for (name, t) in [(scales_name, scales), (zeros_name, zeros)] {
        if t.dtype() != DType::F16 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "flash_attention_fwd_int4_kv requires {name} in F16, got {:?}",
                    t.dtype()
                ),
            });
        }
    }
    let shape = quant.shape().to_vec();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: quant_name,
            reason: format!("expected 4D [B, H, S, D/2], got {}D", shape.len()),
        });
    }
    let (batch, heads, seq_len, packed_head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let head_dim = packed_head_dim * 2;
    let gs = group_size as usize;
    let groups_per_token = head_dim / gs;

    let device = quant.device();
    let per_bh_packed = seq_len * packed_head_dim;
    let per_bh_groups = seq_len * groups_per_token;

    // Extract once: `Tensor::from_slice` below needs plain slices per (batch,
    // head) anyway, and `KvCacheQuantOps::dequantize_kv_int4` is CPU-only
    // (it calls `to_vec` internally too), so this adds no GPU round trip.
    // `dequantize_kv_int4`'s CPU impl reads scales/zeros as raw F32 bytes
    // (`to_vec::<f32>()`), so the F16 inputs are widened to F32 up front —
    // reinterpreting F16 bytes as F32 would silently corrupt them.
    let packed_data = quant.to_vec::<u8>();
    let scale_data: Vec<f32> = scales
        .to_vec::<half::f16>()
        .into_iter()
        .map(|v| v.to_f32())
        .collect();
    let zero_data: Vec<f32> = zeros
        .to_vec::<half::f16>()
        .into_iter()
        .map(|v| v.to_f32())
        .collect();

    let mut out = vec![0.0f32; batch * heads * seq_len * head_dim];

    for b in 0..batch {
        for h in 0..heads {
            let bh = b * heads + h;
            let packed_bh = &packed_data[bh * per_bh_packed..(bh + 1) * per_bh_packed];
            let scales_bh = &scale_data[bh * per_bh_groups..(bh + 1) * per_bh_groups];
            let zeros_bh = &zero_data[bh * per_bh_groups..(bh + 1) * per_bh_groups];

            let packed_t =
                Tensor::<CpuRuntime>::from_slice(packed_bh, &[seq_len, packed_head_dim], device)?;
            let scales_t = Tensor::<CpuRuntime>::from_slice(scales_bh, &[per_bh_groups], device)?;
            let zeros_t = Tensor::<CpuRuntime>::from_slice(zeros_bh, &[per_bh_groups], device)?;

            let dequant_bh = client.dequantize_kv_int4(
                &packed_t,
                &scales_t,
                &zeros_t,
                seq_len,
                head_dim,
                group_size,
                DType::F32,
            )?;
            let dequant_data = dequant_bh.to_vec::<f32>();
            let out_base = bh * seq_len * head_dim;
            out[out_base..out_base + seq_len * head_dim].copy_from_slice(&dequant_data);
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        &[batch, heads, seq_len, head_dim],
        device,
    )?)
}

/// CPU reference for `flash_attention_fwd_int4_kv`: dequantize K/V via the
/// shared INT4 dequant path, then run `standard_attention_fwd`.
/// `num_kv_heads` is fixed to `num_heads` — the kernel this mirrors has no
/// GQA.
#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_int4_kv_impl(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k_quant: &Tensor<CpuRuntime>,
    v_quant: &Tensor<CpuRuntime>,
    k_scales: &Tensor<CpuRuntime>,
    k_zeros: &Tensor<CpuRuntime>,
    v_scales: &Tensor<CpuRuntime>,
    v_zeros: &Tensor<CpuRuntime>,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
    group_size: Int4GroupSize,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    // The CUDA kernel groups int4 elements by `elem_col / group_size` WITHIN
    // a token; `dequantize_kv_int4` (used below) groups by flattened index
    // `i / group_size` over `token*head_dim`. The two conventions agree only
    // when `head_dim % group_size == 0` — otherwise groups straddle token
    // boundaries here but never on CUDA, and the backends silently disagree.
    // Do not remove this check: it is the only thing stopping that.
    let gs = group_size as usize;
    if !head_dim.is_multiple_of(gs) {
        return Err(Error::InvalidArgument {
            arg: "group_size",
            reason: format!(
                "flash_attention_fwd_int4_kv requires head_dim ({head_dim}) to be a multiple \
                 of group_size ({gs}): per-token grouping only agrees with the flattened \
                 grouping dequantize_kv_int4 uses when the group divides head_dim evenly"
            ),
        });
    }

    let k = dequant_int4_kv(
        client, "k_quant", k_quant, "k_scales", k_scales, "k_zeros", k_zeros, group_size,
    )?;
    let v = dequant_int4_kv(
        client, "v_quant", v_quant, "v_scales", v_scales, "v_zeros", v_zeros, group_size,
    )?;
    let cfg = StandardAttnConfig {
        num_heads,
        num_kv_heads: num_heads,
        causal,
        window_size: 0,
    };
    standard_attention_fwd(client, q, &k, &v, cfg)
}

//! Additive attention masks shared by the model architectures.
//!
//! One builder serves every block that needs a causal (optionally
//! sliding-window) additive mask, so the masking rule lives in a single place
//! and cannot drift between architectures.

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, LinalgOps, ScalarOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Additive causal mask, shape `[1, 1, sq, sk]`: `0` where a query may attend,
/// a large negative where it may not.
///
/// Row `i` is absolute position `key_offset + i`, where `key_offset = sk - sq`.
/// A KV-cached decode step passes `sq` new queries against `sk` cached keys, so
/// the offset is the number of keys already in the cache. Prefill/training
/// passes `sq == sk` and the offset is `0`.
///
/// Built as `triu(full(NEG), key_offset + 1)` — `triu` keeps the strictly upper
/// triangle relative to that diagonal (the future) and zeroes the diagonal and
/// below (the past and self), which is exactly the additive form.
///
/// The fill is `f32::MIN`, not `-inf`: `-inf` survives softmax fine, but a
/// `0 * -inf` anywhere in a fused path is NaN, and HuggingFace uses
/// `torch.finfo(dtype).min` for the same reason. Leading shape is `[1, 1]` so it
/// broadcasts across batch and heads instead of materializing `[B, H, S, S]` per
/// layer.
///
/// `window_size > 0` additionally masks keys that fall out of the sliding
/// window: `j + window_size <= key_offset + i`. That is
/// `tril(diagonal = key_offset - window_size)`. The window is INCLUSIVE of the
/// current token, so row `i` keeps `j ∈ (key_offset + i - window_size,
/// key_offset + i]` — exactly `window_size` keys. This matches the kernel
/// contract in `ops/impl_generic/attention/flash_standard.rs`.
///
/// `dtype` is the dtype the mask is returned in — it MUST be the dtype of the
/// attention scores it will be added to, because the additive-mask sites do not
/// reconcile dtypes for it. The triangular construction always runs in F32 and
/// the result is cast once at the end, so a BF16/F16 mask is bit-identical to
/// rounding the F32 mask. `f32::MIN` is representable in BF16 and F16 alike
/// (F16 saturates to its own most-negative finite value), so the fill stays a
/// finite large negative and the `0 * -inf = NaN` hazard stays closed.
///
/// The two masked regions are DISJOINT for every `window_size >= 1`:
/// `triu(key_offset + 1)` keeps `j >= key_offset + i + 1` and
/// `tril(key_offset - window_size)` keeps `j <= key_offset + i - window_size`.
/// No element is covered twice, so summing the two `f32::MIN`-filled tensors
/// never evaluates `MIN + MIN` (which would overflow to `-inf` and reintroduce
/// the `0 * -inf = NaN` hazard the `f32::MIN` fill exists to avoid).
pub fn causal_window_mask<R, C>(
    client: &C,
    sq: usize,
    sk: usize,
    window_size: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    // Exactly the ops this builder calls — NOT the `ModelClient` umbrella.
    // A caller with a small bound list (a trainer's model) must be able to
    // build the same mask without inheriting 17 supertraits.
    C: ScalarOps<R> + LinalgOps<R> + BinaryOps<R>,
    R::Client: TypeConversionOps<R>,
{
    let key_offset = sk.saturating_sub(sq) as i64;
    let zeros = Tensor::<R>::zeros(&[sq, sk], DType::F32, device)?;
    let filled = client.add_scalar(&zeros, f32::MIN as f64)?;
    let future = client.triu(&filled, key_offset + 1)?;
    let masked = if window_size == 0 {
        future
    } else {
        let too_old = client.tril(&filled, key_offset - window_size as i64)?;
        client.add(&future, &too_old)?
    };
    // `to_dtype` is a no-op clone when `dtype` is already F32, so the common
    // case pays nothing for this call.
    Ok(masked.reshape(&[1, 1, sq, sk])?.to_dtype(dtype)?)
}

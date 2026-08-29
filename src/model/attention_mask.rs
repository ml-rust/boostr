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
/// The fill is the target dtype's most-negative finite value ([`mask_fill`]),
/// not `-inf`: `-inf` survives softmax fine, but a
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
/// the result is cast once at the end. The fill is picked from `dtype` by
/// [`mask_fill`] BEFORE the build, so the cast is exact and the fill stays a
/// finite large negative in every dtype — see [`mask_fill`] for why casting
/// `f32::MIN` down instead would produce `-inf`.
///
/// The two masked regions are DISJOINT for every `window_size >= 1`:
/// `triu(key_offset + 1)` keeps `j >= key_offset + i + 1` and
/// `tril(key_offset - window_size)` keeps `j <= key_offset + i - window_size`.
/// No element is covered twice, so summing the two fill-valued tensors
/// never evaluates `MIN + MIN` (which would overflow to `-inf` and reintroduce
/// the `0 * -inf = NaN` hazard the finite fill exists to avoid).
/// The most-negative FINITE value of `dtype`, as the F32 fill the mask is built
/// with.
///
/// `f32::MIN` overflows both half formats: `bf16::from_f32(f32::MIN)` and
/// `f16::from_f32(f32::MIN)` are both `-inf`, not a saturated finite value
/// (measured, not assumed). Filling in F32 and casting down would therefore
/// undo exactly what the `f32::MIN` fill exists to guarantee — see
/// [`causal_window_mask`]'s note on the `0 * -inf = NaN` hazard. Choosing the
/// fill from the TARGET dtype up front keeps the cast exact: every value here
/// is representable in F32 and in `dtype` alike.
fn mask_fill(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => f64::from(half::f16::MIN.to_f32()),
        DType::BF16 => f64::from(half::bf16::MIN.to_f32()),
        // F32, F64 and the integer dtypes all hold `f32::MIN` exactly. A mask
        // is never built in an integer dtype (it is added to attention scores),
        // so no narrower case is reachable.
        _ => f64::from(f32::MIN),
    }
}

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
    let filled = client.add_scalar(&zeros, mask_fill(dtype))?;
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

#[cfg(test)]
mod tests {
    use super::{causal_window_mask, mask_fill};
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;

    /// The whole point of the fill: it must stay FINITE after the cast into
    /// the mask's own dtype. `f32::MIN` does not — casting it to either half
    /// format yields `-inf` — so a regression here is a silent reintroduction
    /// of the `0 * -inf = NaN` hazard, invisible to shape checks.
    #[test]
    fn fill_is_finite_in_every_float_dtype() {
        assert!(half::f16::from_f64(mask_fill(DType::F16)).is_finite());
        assert!(half::bf16::from_f64(mask_fill(DType::BF16)).is_finite());
        assert!((mask_fill(DType::F32) as f32).is_finite());
        // The guard this exists to hold: the naive fill really does overflow.
        assert!(!half::f16::from_f32(f32::MIN).is_finite());
        assert!(!half::bf16::from_f32(f32::MIN).is_finite());
    }

    /// A built BF16 mask carries finite masked entries and exact zeros on the
    /// admitted side, so the F32 build + single cast survives the round trip.
    // BF16 casts need numr's `f16` feature; without it `to_dtype` errors.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_mask_is_finite_and_causal() {
        let (client, device) = cpu_setup();
        let (sq, sk) = (4usize, 4usize);
        let mask =
            causal_window_mask::<CpuRuntime, _>(&client, sq, sk, 0, DType::BF16, &device).unwrap();
        assert_eq!(mask.dtype(), DType::BF16);
        let values: Vec<f32> = mask.to_dtype(DType::F32).unwrap().to_vec();
        for i in 0..sq {
            for j in 0..sk {
                let v = values[i * sk + j];
                assert!(v.is_finite(), "row {i} col {j} is {v}");
                if j <= i {
                    assert_eq!(v, 0.0, "past/self at row {i} col {j} must be admitted");
                } else {
                    assert!(v < -1.0e30, "future at row {i} col {j} must be masked");
                }
            }
        }
    }
}

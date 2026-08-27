//! The prefill/training additive attention mask (ALiBi bias or causal+window).

use super::spec::AttentionCoreSpec;
use crate::error::Result;
use crate::model::attention_mask::causal_window_mask;
use crate::ops::traits::position::alibi::AlibiOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, LinalgOps, ScalarOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The prefill/training additive mask: ALiBi bias or causal(+window).
///
/// Split out so the masking rule is directly testable — a non-causal mask here
/// is invisible to shape checks and still produces fluent text.
///
/// `dtype` is the dtype the mask is returned in; it MUST match the attention
/// scores it will be added to. See [`causal_window_mask`] for why the build
/// runs in F32 and casts once.
///
/// Row `i` is at ABSOLUTE position `sk - sq + i`, matching
/// `causal_window_mask` and `flash_standard::build_attention_mask`. Prefill and
/// training pass `sq == sk`, giving offset `0`.
pub fn prefill_attention_mask<R, C>(
    client: &C,
    batch: usize,
    sq: usize,
    sk: usize,
    spec: &AttentionCoreSpec<'_, R>,
    dtype: DType,
    device: &R::Device,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R> + LinalgOps<R> + BinaryOps<R> + AlibiOps<R>,
    R::Client: TypeConversionOps<R>,
{
    Ok(if spec.use_alibi {
        // MUST be the `_causal` kernel. `alibi_add_bias` writes a SYMMETRIC
        // `-slope * |qi - ki|` over the whole rectangle and masks nothing,
        // so using it here let every prefill position attend to FUTURE
        // tokens — the same defect the non-ALiBi path had until parity
        // testing caught it. `alibi_add_bias_causal` adds the same distance
        // bias and sets `ki > qi + position` to -inf.
        //
        // `position` is the absolute position of query row `0`, i.e. the
        // number of keys that precede the query block. Prefill/training has
        // `sq == sk` and `position == 0`.
        let mask = Tensor::<R>::zeros(&[batch, spec.num_heads, sq, sk], DType::F32, device)?;
        client.alibi_add_bias_causal(
            &mask,
            batch,
            spec.num_heads,
            sq,
            sk,
            sk.saturating_sub(sq),
        )?;
        // `alibi_add_bias_causal` writes F32 slopes, so the bias is built in
        // F32 and cast once here, exactly as `causal_window_mask` does.
        Var::new(mask.to_dtype(dtype)?, false)
    } else {
        // `causal_window_mask` owns causality, the inclusive sliding window,
        // the `sk - sq` key offset, and the `f32::MIN` fill for every
        // architecture.
        Var::new(
            causal_window_mask::<R, C>(client, sq, sk, spec.sliding_window, dtype, device)?,
            false,
        )
    })
}

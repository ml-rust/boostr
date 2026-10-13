//! The cached layer stack shared by `prefill` and `decode_step`.
//!
//! # Numerical agreement
//!
//! The step-wise and full-sequence paths run the same weights, the same
//! precomputed RoPE tables at the same absolute positions, and select the same
//! keys — but reduce over the key axis in a different order (one row at a time
//! versus a full matrix). They agree to roughly `1e-4` absolute, not bitwise.
//! The reference's own two paths differ by `9.9e-5` for the same reason.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::audio::voxcpm::minicpm4::cache_step::CacheStep;
use crate::model::audio::voxcpm::minicpm4::model::MiniCpm4Model;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// The cached layer stack: `[batch, seq, hidden]` covering the absolute
    /// positions `step` gives each row -> `[batch, seq, hidden]` after the
    /// final `norm`.
    ///
    /// Same layer order and same final norm as
    /// [`forward`](MiniCpm4Model::forward); only attention differs. Prefill
    /// (`seq == prefix`, position 0) and a decode step (`seq == 1`) both run
    /// through here, so the two cached shapes cannot drift apart.
    pub(super) fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        step: &CacheStep<'_, R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        // Deferred-residual fusion across layers: each layer folds the
        // PREVIOUS layer's MLP output into its own input norm instead of a
        // separate add, and hands its own MLP output on unadded. See
        // `MiniCpm4Layer::forward_with_pending_residual`.
        let mut h = x.clone();
        let mut pending: Option<Var<R>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            let (new_h, mlp_out) = layer.forward_cached_with_pending_residual(
                client,
                &h,
                pending.as_ref(),
                self.rope.as_ref(),
                cache,
                step,
            )?;
            h = new_h;
            pending = Some(mlp_out);
        }
        match pending {
            Some(last_mlp) => Ok(self.norm.residual_norm(client, &h, &last_mlp)?.0),
            // `self.layers` is empty: nothing was deferred.
            None => self.norm.forward(client, &h),
        }
    }
}

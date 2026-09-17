//! The cached layer stack shared by `prefill` and `decode_step`.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
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
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// The cached layer stack: `[batch, seq, hidden]` covering absolute
    /// positions `position..position + seq` -> `[batch, seq, hidden]` after the
    /// final `norm`.
    ///
    /// Same layer order and same final norm as
    /// [`forward`](MiniCpm4Model::forward); only attention differs. Prefill
    /// (`seq == prefix`, `position == 0`) and a decode step (`seq == 1`) both
    /// run through here, so the two cached shapes cannot drift apart.
    pub(super) fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
        kv_start: Option<&Tensor<R>>,
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
                position,
                kv_start,
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

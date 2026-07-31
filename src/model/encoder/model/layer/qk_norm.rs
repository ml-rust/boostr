//! Query/key normalisation, applied before RoPE.
//!
//! Two families normalise Q and K, over different axes, with different norm
//! types. Both hooks are called unconditionally from the attention paths and
//! each is a no-op for the scope it does not own, so adding an architecture
//! cannot leave one of the two attention paths behind.

use super::encoder_layer::EncoderLayer;
use crate::error::Result;
use crate::model::encoder::config::QkNormScope;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    /// QK-norm over the whole `hidden_size` projection output.
    ///
    /// Call with Q and K still shaped `[.., hidden]`, i.e. BEFORE the reshape
    /// into heads. A no-op unless `qk_norm_scope` is [`QkNormScope::Hidden`].
    pub(super) fn qk_norm_hidden<C>(
        &self,
        client: &C,
        q: Var<R>,
        k: Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        if self.qk_norm_scope != QkNormScope::Hidden {
            return Ok((q, k));
        }
        self.apply_qk_norm(client, q, k)
    }

    /// QK-norm over `head_dim`, per head.
    ///
    /// Call with Q and K already reshaped so `head_dim` is the last axis —
    /// `[B, H, S, D]` on the padded path, `[total_tokens, H, D]` on the packed
    /// one. A no-op unless `qk_norm_scope` is [`QkNormScope::PerHead`].
    pub(super) fn qk_norm_per_head<C>(
        &self,
        client: &C,
        q: Var<R>,
        k: Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        if self.qk_norm_scope != QkNormScope::PerHead {
            return Ok((q, k));
        }
        self.apply_qk_norm(client, q, k)
    }

    fn apply_qk_norm<C>(&self, client: &C, q: Var<R>, k: Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let q = match &self.q_norm {
            Some(n) => n.forward(client, &q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(client, &k)?,
            None => k,
        };
        Ok((q, k))
    }
}

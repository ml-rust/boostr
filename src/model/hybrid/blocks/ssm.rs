//! The hybrid model's SSM block: pre-norm, Mamba2, residual.

use crate::error::{Error, Result};
use crate::inference::SsmState;
use crate::model::mamba::mamba2::Mamba2;
use crate::model::traits::ModelClient;
use crate::nn::RmsNorm;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps,
    TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// SSM block: pre-norm → Mamba2 → residual
pub(in crate::model::hybrid) struct SsmBlock<R: Runtime> {
    pub(in crate::model::hybrid) norm: RmsNorm<R>,
    pub(in crate::model::hybrid) mamba: Mamba2<R>,
}

impl<R: Runtime<DType = DType>> SsmBlock<R> {
    pub(in crate::model::hybrid) fn forward_inference<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: &mut SsmState<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        let normed = self.norm.forward(client, x)?;
        let out_tensor = self
            .mamba
            .forward_inference(client, normed.tensor(), state)?;
        let out = Var::new(out_tensor, false);
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
    }
}

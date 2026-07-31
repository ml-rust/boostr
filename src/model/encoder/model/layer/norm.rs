//! Norm layer variant: LayerNorm (BERT/NomicBert) or RmsNorm (Gemma/Qwen3).

use crate::error::Result;
use crate::nn::{LayerNorm, RmsNorm};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Norm layer that can be either a LayerNorm (BERT/NomicBert) or RmsNorm (Gemma/Qwen3).
pub(in crate::model::encoder) enum NormLayer<R: Runtime> {
    LayerNorm(LayerNorm<R>),
    RmsNorm(RmsNorm<R>),
}

impl<R: Runtime<DType = DType>> NormLayer<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::LayerNorm(ln) => ln.forward(client, x),
            Self::RmsNorm(rn) => rn.forward(client, x),
        }
    }
}

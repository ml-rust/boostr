//! LLaMA SwiGLU MLP block.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::MaybeQuantLinear;
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct LlamaMlp<R: Runtime> {
    pub(crate) gate_proj: MaybeQuantLinear<R>,
    pub(crate) up_proj: MaybeQuantLinear<R>,
    pub(crate) down_proj: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> LlamaMlp<R> {
    /// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
    {
        // Try fused SwiGLU path: single kernel for silu(gate_proj(x)) * up_proj(x)
        if let (MaybeQuantLinear::Quantized(gate_ql), MaybeQuantLinear::Quantized(up_ql)) =
            (&self.gate_proj, &self.up_proj)
        {
            if gate_ql.bias().is_none() && up_ql.bias().is_none() {
                let hidden_t = client.quant_swiglu(x.tensor(), gate_ql.weight(), up_ql.weight())?;
                let hidden = Var::new(hidden_t, false);
                return self.down_proj.forward(client, &hidden);
            }
        }

        // Fallback: batched gate+up + separate silu_mul
        let gate_up =
            MaybeQuantLinear::forward_batch(&[&self.gate_proj, &self.up_proj], client, x)?;
        let (gate, up) = (&gate_up[0], &gate_up[1]);

        let hidden = var_silu_mul(gate, up, client).map_err(Error::Numr)?;
        let result = self.down_proj.forward(client, &hidden)?;
        Ok(result)
    }
}

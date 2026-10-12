//! LLaMA-path MLP block: gated (SwiGLU, GeGLU) or plain (GELU).

use crate::error::{Error, Result};
use crate::model::config::MlpActivation;
use crate::model::traits::ModelClient;
use crate::nn::MaybeQuantLinear;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_gelu_erf_mul, var_gelu_mul, var_silu, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `down(act(gate(x)) * up(x))` with a gate, `down(act(up(x)))` without.
pub struct LlamaMlp<R: Runtime> {
    /// `None` for a plain MLP (GPT-NeoX, Falcon, StarCoder2).
    pub(crate) gate_proj: Option<MaybeQuantLinear<R>>,
    pub(crate) up_proj: MaybeQuantLinear<R>,
    pub(crate) down_proj: MaybeQuantLinear<R>,
    pub(crate) activation: MlpActivation,
}

impl<R: Runtime<DType = DType>> LlamaMlp<R> {
    /// A gated SiLU MLP, the Llama MLP.
    pub fn swiglu(
        gate_proj: MaybeQuantLinear<R>,
        up_proj: MaybeQuantLinear<R>,
        down_proj: MaybeQuantLinear<R>,
    ) -> Self {
        Self {
            gate_proj: Some(gate_proj),
            up_proj,
            down_proj,
            activation: MlpActivation::Silu,
        }
    }

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
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let Some(gate_proj) = &self.gate_proj else {
            let up = self.up_proj.forward(client, x)?;
            let hidden = self.plain_activation(client, &up)?;
            return self.down_proj.forward(client, &hidden);
        };

        // Fused SwiGLU kernel: one pass for silu(gate_proj(x)) * up_proj(x).
        if self.activation == MlpActivation::Silu
            && let (MaybeQuantLinear::Quantized(gate_ql), MaybeQuantLinear::Quantized(up_ql)) =
                (gate_proj, &self.up_proj)
            && gate_ql.bias().is_none()
            && up_ql.bias().is_none()
        {
            let hidden_t = client.quant_swiglu(x.tensor(), gate_ql.weight(), up_ql.weight())?;
            let hidden = Var::new(hidden_t, false);
            return self.down_proj.forward(client, &hidden);
        }

        // Batched gate+up, then the fused activation-mul.
        let gate_up = MaybeQuantLinear::forward_batch(&[gate_proj, &self.up_proj], client, x)?;
        let (gate, up) = (&gate_up[0], &gate_up[1]);
        let hidden = match self.activation {
            MlpActivation::Silu => var_silu_mul(gate, up, client),
            MlpActivation::GeluTanh => var_gelu_mul(gate, up, client),
            MlpActivation::GeluErf => var_gelu_erf_mul(gate, up, client),
        }
        .map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }

    /// `act(x)` for a plain MLP. With a gradient, the GELU forms run the
    /// fused activation-mul against ones, which carries a backward.
    fn plain_activation<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        if self.activation == MlpActivation::Silu {
            return var_silu(x, client).map_err(Error::Numr);
        }
        if !x.requires_grad() {
            let t = if self.activation == MlpActivation::GeluTanh {
                client.gelu(x.tensor())
            } else {
                client.gelu_erf(x.tensor())
            }
            .map_err(Error::Numr)?;
            return Ok(Var::new(t, false));
        }
        let ones = Var::new(
            Tensor::<R>::ones(x.shape(), x.tensor().dtype(), x.tensor().device())?,
            false,
        );
        if self.activation == MlpActivation::GeluTanh {
            var_gelu_mul(x, &ones, client)
        } else {
            var_gelu_erf_mul(x, &ones, client)
        }
        .map_err(Error::Numr)
    }
}

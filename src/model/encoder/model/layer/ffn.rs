//! Feed-forward network variants: standard, SwiGLU, GeGLU.

use super::encoder_layer::EncoderLayer;
use crate::error::{Error, Result};
use crate::model::encoder::config::{FfnVariant, HiddenAct};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_mul, var_silu};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    pub(super) fn ffn<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        match self.ffn_variant {
            FfnVariant::Standard => {
                let h = self.ffn_up.forward(client, x)?;
                let h = match self.hidden_act {
                    HiddenAct::Gelu => {
                        Var::new(client.gelu(h.tensor()).map_err(Error::Numr)?, false)
                    }
                    HiddenAct::Relu => {
                        Var::new(client.relu(h.tensor()).map_err(Error::Numr)?, false)
                    }
                };
                self.ffn_down.forward(client, &h)
            }
            // SwiGLU: ffn_down(silu(ffn_gate(x)) * ffn_up(x))
            FfnVariant::GatedSilu => {
                let gate = self.gate(client, x)?;
                let up = self.ffn_up.forward(client, x)?;
                let gate_act = var_silu(&gate, client).map_err(Error::Numr)?;
                let hidden = var_mul(&gate_act, &up, client).map_err(Error::Numr)?;
                self.ffn_down.forward(client, &hidden)
            }
            // GeGLU: ffn_down(gelu(ffn_gate(x)) * ffn_up(x))
            FfnVariant::GatedGelu => {
                let gate = self.gate(client, x)?;
                let up = self.ffn_up.forward(client, x)?;
                let gate_act = Var::new(client.gelu(gate.tensor()).map_err(Error::Numr)?, false);
                let hidden = var_mul(&gate_act, &up, client).map_err(Error::Numr)?;
                self.ffn_down.forward(client, &hidden)
            }
        }
    }

    /// Gate projection for the gated FFN variants.
    fn gate<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        self.ffn_gate
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "ffn_gate is required by the {:?} FFN variant",
                    self.ffn_variant
                ),
            })?
            .forward(client, x)
    }
}

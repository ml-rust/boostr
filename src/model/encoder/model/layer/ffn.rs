//! Feed-forward network variants: standard, SwiGLU, GeGLU.

use super::encoder_layer::EncoderLayer;
use crate::error::{Error, Result};
use crate::model::encoder::config::FfnVariant;
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
                let h = Var::new(
                    self.hidden_act
                        .forward(client, h.tensor())
                        .map_err(Error::Numr)?,
                    false,
                );
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
            // GeGLU: ffn_down(act(ffn_gate(x)) * ffn_up(x)), act = erf or tanh GELU
            // per `self.hidden_act` (jina-v2 is erf, gemma is tanh).
            FfnVariant::GatedGelu => {
                let gate = self.gate(client, x)?;
                let up = self.ffn_up.forward(client, x)?;
                let gate_act = Var::new(
                    self.hidden_act
                        .forward(client, gate.tensor())
                        .map_err(Error::Numr)?,
                    false,
                );
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

#[cfg(test)]
mod tests {
    use crate::nn::Activation;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, var_mul};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// GeGLU gate math for jina-v2 (`Activation::Gelu`, erf) and gemma
    /// (`Activation::GeluTanh`) on the same fixed gate/up projections. Both
    /// GGUF configs set `ffn_variant: GatedGelu`; only `hidden_act` differs.
    /// The two families must produce different hidden activations, or the
    /// GGUF config bug (one `HiddenAct::Gelu` arm serving both) is back.
    #[test]
    fn jina_v2_and_gemma_geglu_gates_diverge_on_the_same_input() {
        let (client, device) = cpu_setup();
        let gate =
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -0.8, 2.2, 0.3], &[4], &device).unwrap();
        let up = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();

        let jina_v2_act = Activation::Gelu.forward(&client, &gate).unwrap();
        let gemma_act = Activation::GeluTanh.forward(&client, &gate).unwrap();

        let jina_v2_hidden = var_mul(
            &Var::new(jina_v2_act, false),
            &Var::new(up.clone(), false),
            &client,
        )
        .unwrap();
        let gemma_hidden =
            var_mul(&Var::new(gemma_act, false), &Var::new(up, false), &client).unwrap();

        let jina_v2_data: Vec<f32> = jina_v2_hidden.tensor().to_vec();
        let gemma_data: Vec<f32> = gemma_hidden.tensor().to_vec();

        let max_abs_diff = jina_v2_data
            .iter()
            .zip(gemma_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff > 1e-4,
            "jina-v2 (erf) and gemma (tanh) GeGLU gates must diverge: \
             jina_v2={jina_v2_data:?}, gemma={gemma_data:?}"
        );
    }
}

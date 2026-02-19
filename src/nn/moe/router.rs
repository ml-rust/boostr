//! MoE Router — top-k expert gating with load balancing

use crate::error::{Error, Result};
use crate::nn::Linear;
use numr::autograd::{Var, var_softmax};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Router configuration
pub struct MoeRouterConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to route each token to
    pub top_k: usize,
}

/// MoE Router with top-k gating.
///
/// Routes tokens to experts using a learned gate projection.
pub struct MoeRouter<R: Runtime> {
    gate: Linear<R>,
    config: MoeRouterConfig,
}

/// Router output: selected expert indices, weights, and auxiliary loss
pub struct RouterOutput<R: Runtime> {
    /// Expert weights per token: `[batch * seq, top_k]`
    pub weights: Var<R>,
    /// Expert indices per token: `[batch * seq, top_k]` (I64 tensor)
    pub indices: Tensor<R>,
    /// Load balancing auxiliary loss (scalar)
    pub aux_loss: Var<R>,
}

impl<R: Runtime> MoeRouter<R> {
    pub fn new(gate: Linear<R>, config: MoeRouterConfig) -> Self {
        Self { gate, config }
    }

    /// Create from gate weight tensor `[num_experts, hidden_size]`
    pub fn from_tensor(gate_weight: Tensor<R>, config: MoeRouterConfig, trainable: bool) -> Self {
        Self {
            gate: Linear::new(gate_weight, None, trainable),
            config,
        }
    }

    pub fn config(&self) -> &MoeRouterConfig {
        &self.config
    }

    pub fn gate(&self) -> &Linear<R> {
        &self.gate
    }

    /// Route tokens to experts.
    ///
    /// Input: `[num_tokens, hidden_size]`
    /// Returns: RouterOutput with weights, indices, and aux_loss
    pub fn route<C>(&self, client: &C, x: &Var<R>) -> Result<RouterOutput<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + SortingOps<R>
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
    {
        // gate logits: [num_tokens, num_experts]
        let logits = self.gate.forward(client, x)?;

        // softmax over experts dim
        let probs = var_softmax(&logits, -1, client).map_err(Error::Numr)?;

        // top-k selection via numr's SortingOps (stays on device)
        let probs_tensor = probs.tensor();
        let (top_values, top_indices) = client
            .topk(probs_tensor, self.config.top_k, -1, true, true)
            .map_err(Error::Numr)?;

        // Normalize top-k weights to sum to 1
        let weight_sum = client.sum(&top_values, &[1], true)?;
        let normalized_weights = client.div(&top_values, &weight_sum)?;

        // Compute load balancing auxiliary loss
        let aux_loss = self.compute_aux_loss(client, &probs, &top_indices)?;

        Ok(RouterOutput {
            weights: Var::new(normalized_weights, probs.requires_grad()),
            indices: top_indices,
            aux_loss,
        })
    }

    /// Compute load balancing auxiliary loss.
    ///
    /// Loss = num_experts * sum(P_e * N_e) where:
    /// - P_e = mean probability assigned to expert e across all tokens
    /// - N_e = fraction of tokens routed to expert e
    ///
    /// Uses bincount for on-device N_e computation (no CPU transfers).
    fn compute_aux_loss<C>(&self, client: &C, probs: &Var<R>, indices: &Tensor<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
    {
        let probs_tensor = probs.tensor();
        let num_tokens = probs_tensor.shape()[0];
        let num_experts = self.config.num_experts;
        let k = self.config.top_k;

        // P_e: mean probability per expert [num_experts]
        let p_e = client.mean(probs_tensor, &[0], false)?;

        // N_e: fraction of tokens routed to each expert (on-device via bincount)
        let flat_indices = indices.reshape(&[indices.numel()]).map_err(Error::Numr)?;
        let counts = client
            .bincount(&flat_indices, None, num_experts)
            .map_err(Error::Numr)?;
        let counts_f32 = client.cast(&counts, DType::F32).map_err(Error::Numr)?;
        let total = (num_tokens * k) as f64;
        let n_e = client.div_scalar(&counts_f32, total)?;

        // aux_loss = num_experts * sum(P_e * N_e)
        let p_e_var = Var::new(p_e, probs.requires_grad());
        let n_e_var = Var::new(n_e, false);

        let pn = numr::autograd::var_mul(&p_e_var, &n_e_var, client).map_err(Error::Numr)?;
        let loss_sum = numr::autograd::var_sum(&pn, &[0], false, client).map_err(Error::Numr)?;
        let loss = numr::autograd::var_mul_scalar(&loss_sum, num_experts as f64, client)
            .map_err(Error::Numr)?;

        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_router_output_shapes() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let num_experts = 4;
        let top_k = 2;

        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 16], &[num_experts, hidden], &device);

        let config = MoeRouterConfig { num_experts, top_k };
        let router = MoeRouter::from_tensor(gate_w, config, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device),
            false,
        );
        let output = router.route(&client, &input).unwrap();

        assert_eq!(output.weights.shape(), &[3, top_k]);
        assert_eq!(output.indices.shape(), &[3, top_k]);
        assert_eq!(output.aux_loss.tensor().numel(), 1);
    }
}

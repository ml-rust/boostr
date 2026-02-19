//! MoE Layer — combines router with expert MLPs

use crate::error::{Error, Result};
use crate::nn::moe::expert::Expert;
use crate::nn::moe::router::{MoeRouter, RouterOutput};
use numr::autograd::{Var, var_add, var_narrow};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// MoE layer configuration
pub struct MoeLayerConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Top-k experts per token
    pub top_k: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size per expert
    pub intermediate_size: usize,
}

/// Output from MoE layer forward pass
pub struct MoeOutput<R: Runtime> {
    /// Layer output: `[num_tokens, hidden_size]`
    pub output: Var<R>,
    /// Router auxiliary loss for load balancing
    pub aux_loss: Var<R>,
}

/// Mixture of Experts layer.
///
/// Routes tokens to top-k experts, computes expert outputs,
/// and returns the weighted combination.
///
/// All computation stays on-device — no GPU-CPU transfers.
pub struct MoeLayer<R: Runtime> {
    router: MoeRouter<R>,
    experts: Vec<Expert<R>>,
    /// Optional shared expert (always active for all tokens)
    shared_expert: Option<Expert<R>>,
}

impl<R: Runtime> MoeLayer<R> {
    pub fn new(
        router: MoeRouter<R>,
        experts: Vec<Expert<R>>,
        shared_expert: Option<Expert<R>>,
    ) -> Self {
        Self {
            router,
            experts,
            shared_expert,
        }
    }

    pub fn router(&self) -> &MoeRouter<R> {
        &self.router
    }

    pub fn experts(&self) -> &[Expert<R>] {
        &self.experts
    }

    pub fn shared_expert(&self) -> Option<&Expert<R>> {
        self.shared_expert.as_ref()
    }

    /// Forward pass with auxiliary loss.
    ///
    /// Input: `[num_tokens, hidden_size]`
    /// Returns: MoeOutput with output tensor and aux_loss
    ///
    /// Strategy: iterate over experts (not tokens). For each expert,
    /// compute output for all tokens, then mask-and-weight by routing decisions.
    /// All ops stay on-device.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<MoeOutput<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + SortingOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
    {
        let num_tokens = x.shape()[0];
        let hidden_size = x.shape()[1];
        let top_k = self.router.config().top_k;

        // Route tokens to experts
        let RouterOutput {
            weights,
            indices,
            aux_loss,
        } = self.router.route(client, x)?;

        // Initialize output accumulator as zeros
        let mut output = Var::new(
            Tensor::<R>::zeros(&[num_tokens, hidden_size], DType::F32, x.tensor().device()),
            x.requires_grad(),
        );

        // For each top-k slot, process all tokens through their assigned expert
        for k_idx in 0..top_k {
            // Extract indices and weights for this slot: [num_tokens, 1]
            let slot_indices =
                var_narrow(&Var::new(indices.clone(), false), -1, k_idx, 1).map_err(Error::Numr)?;
            let slot_weights = var_narrow(&weights, -1, k_idx, 1).map_err(Error::Numr)?;

            // For each expert, find which tokens are routed to it and process them
            for (expert_idx, expert) in self.experts.iter().enumerate() {
                // Create mask: slot_indices == expert_idx (on-device)
                // Build constant tensor via ones * scalar
                let expert_id_tensor = {
                    let ones = Tensor::<R>::ones(
                        slot_indices.shape(),
                        slot_indices.tensor().dtype(),
                        x.tensor().device(),
                    );
                    client
                        .mul_scalar(&ones, expert_idx as f64)
                        .map_err(Error::Numr)?
                };
                let mask = client
                    .eq(slot_indices.tensor(), &expert_id_tensor)
                    .map_err(Error::Numr)?;
                let mask_f32 = client.cast(&mask, DType::F32).map_err(Error::Numr)?;

                // Check if any tokens are routed to this expert
                let count = client.sum(&mask_f32, &[0, 1], false)?;
                let count_val: Vec<f32> = count.to_vec();
                if count_val[0] < 0.5 {
                    continue;
                }

                // Run expert on ALL tokens, then mask out unrouted ones
                let expert_out = expert.forward(client, x)?;

                // Weight: expert_out * slot_weight * mask
                // mask: [num_tokens, 1], slot_weights: [num_tokens, 1]
                let mask_var = Var::new(mask_f32, false);
                let weighted = numr::autograd::var_mul(&expert_out, &slot_weights, client)
                    .map_err(Error::Numr)?;
                let masked =
                    numr::autograd::var_mul(&weighted, &mask_var, client).map_err(Error::Numr)?;

                output = var_add(&output, &masked, client).map_err(Error::Numr)?;
            }
        }

        // Add shared expert if present
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(client, x)?;
            output = var_add(&output, &shared_out, client).map_err(Error::Numr)?;
        }

        Ok(MoeOutput { output, aux_loss })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::moe::router::MoeRouterConfig;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_moe_layer_forward_shape() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let inter = 8;
        let num_experts = 2;
        let top_k = 1;

        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device);
        let config = MoeRouterConfig { num_experts, top_k };
        let router = MoeRouter::from_tensor(gate_w, config, false);

        let experts: Vec<Expert<CpuRuntime>> = (0..num_experts)
            .map(|_| {
                let gw = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[inter, hidden], &device);
                let uw = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[inter, hidden], &device);
                let dw = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[hidden, inter], &device);
                Expert::from_tensors(gw, uw, dw, false)
            })
            .collect();

        let layer = MoeLayer::new(router, experts, None);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device),
            false,
        );
        let result = layer.forward(&client, &input).unwrap();

        assert_eq!(result.output.shape(), &[3, hidden]);
    }
}

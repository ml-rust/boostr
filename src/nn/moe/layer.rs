//! MoE Layer — combines router with expert MLPs

use crate::error::{Error, Result};
use crate::nn::loss::router_z_loss;
use crate::nn::moe::expert::Expert;
use crate::nn::moe::router::{MoeRouter, RouterOutput};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_add, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TensorOps, TypeConversionOps, UnaryOps,
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
    /// ST-MoE router z-loss computed from tracked pre-softmax router logits.
    pub z_loss: Var<R>,
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
    /// Weight applied to the shared expert's output before it is added.
    shared_expert_scale: f32,
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
            shared_expert_scale: 1.0,
        }
    }

    /// Scale the shared expert's contribution before summing it with the
    /// routed experts.
    ///
    /// The shared expert is active for every token while the routed experts
    /// are not, so adding it unscaled changes the magnitude of the layer
    /// output relative to a top-k-only mixture. A common choice is
    /// `1 / (top_k + 1)`, which averages the shared expert with the `top_k`
    /// routed ones instead of letting it dominate.
    ///
    /// Defaults to `1.0` (add unscaled).
    pub fn with_shared_expert_scale(mut self, scale: f32) -> Result<Self> {
        if !scale.is_finite() {
            return Err(Error::InvalidArgument {
                arg: "shared_expert_scale",
                reason: format!("expected a finite value, got {scale}"),
            });
        }
        self.shared_expert_scale = scale;
        Ok(self)
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

    pub fn shared_expert_scale(&self) -> f32 {
        self.shared_expert_scale
    }

    /// Forward pass with auxiliary loss.
    ///
    /// Input: `[num_tokens, hidden_size]`
    /// Returns: MoeOutput with output tensor, aux_loss, and z_loss
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
            + CompareOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + DequantOps<R>,
    {
        let num_tokens = x.shape()[0];
        let hidden_size = x.shape()[1];
        let top_k = self.router.config().top_k;

        // Route tokens to experts
        let RouterOutput {
            weights,
            indices,
            logits,
            aux_loss,
        } = self.router.route(client, x)?;
        let z_logits = if logits.shape().len() == 2 {
            logits.clone()
        } else {
            let logits_shape = logits.shape().to_vec();
            let num_experts = *logits_shape.last().ok_or_else(|| Error::InvalidArgument {
                arg: "router_logits",
                reason: "expected at least one dimension".into(),
            })?;
            let num_tokens = logits_shape[..logits_shape.len() - 1].iter().product();
            var_reshape(&logits, &[num_tokens, num_experts]).map_err(Error::Numr)?
        };
        let z_loss = router_z_loss(client, &z_logits)?;

        // Initialize output accumulator as zeros
        let mut output = Var::new(
            Tensor::<R>::zeros(&[num_tokens, hidden_size], DType::F32, x.tensor().device())?,
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
                    )?;
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

        // Add shared expert if present, weighted by shared_expert_scale.
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(client, x)?;
            let shared_out = if self.shared_expert_scale == 1.0 {
                shared_out
            } else {
                numr::autograd::var_mul_scalar(&shared_out, self.shared_expert_scale as f64, client)
                    .map_err(Error::Numr)?
            };
            output = var_add(&output, &shared_out, client).map_err(Error::Numr)?;
        }

        Ok(MoeOutput {
            output,
            aux_loss,
            z_loss,
        })
    }
}

#[cfg(test)]
mod tests;

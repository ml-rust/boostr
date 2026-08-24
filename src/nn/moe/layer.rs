//! MoE Layer — combines router with expert MLPs

use crate::error::{Error, Result};
use crate::nn::loss::router_z_loss;
use crate::nn::moe::expert::Expert;
use crate::nn::moe::router::{MoeRouter, RouterOutput};
use numr::autograd::{Var, var_add, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TensorOps, UnaryOps,
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
            + UnaryOps<R>,
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + UnaryOps<R>,
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
            Tensor::<R>::try_zeros(&[num_tokens, hidden_size], DType::F32, x.tensor().device())?,
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
                    let ones = Tensor::<R>::try_ones(
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
mod tests {
    use super::*;
    use crate::nn::moe::router::MoeRouterConfig;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn experts(
        num_experts: usize,
        hidden: usize,
        inter: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Vec<Expert<CpuRuntime>> {
        (0..num_experts)
            .map(|expert_idx| {
                let scale = 0.05f32 + expert_idx as f32 * 0.02;
                let gw = Tensor::<CpuRuntime>::from_slice(
                    &vec![scale; inter * hidden],
                    &[inter, hidden],
                    device,
                );
                let uw = Tensor::<CpuRuntime>::from_slice(
                    &vec![scale + 0.01; inter * hidden],
                    &[inter, hidden],
                    device,
                );
                let dw = Tensor::<CpuRuntime>::from_slice(
                    &vec![scale - 0.01; hidden * inter],
                    &[hidden, inter],
                    device,
                );
                Expert::from_tensors(gw, uw, dw, false)
            })
            .collect()
    }

    #[test]
    fn test_moe_layer_forward_shape() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let inter = 8;
        let num_experts = 2;
        let top_k = 1;

        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device);
        let config = MoeRouterConfig::new(num_experts, top_k);
        let router = MoeRouter::from_tensor(gate_w, config, false);

        let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device),
            false,
        );
        let result = layer.forward(&client, &input).unwrap();

        assert_eq!(result.output.shape(), &[3, hidden]);
        assert_eq!(result.z_loss.tensor().numel(), 1);
    }

    /// The shared expert is active for every token, so its contribution must be
    /// weightable. Scaling by `s` must shift the output by exactly `(s - 1)`
    /// times the shared expert's own output — not merely "change it".
    #[test]
    fn shared_expert_scale_weights_the_shared_contribution() {
        fn run(scale: Option<f32>) -> Vec<f32> {
            let (client, device) = cpu_setup();
            let (hidden, inter, num_experts, top_k) = (4, 8, 2, 1);

            let gate_w =
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device);
            let router =
                MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), false);
            let shared = experts(1, hidden, inter, &device).pop();

            let mut layer =
                MoeLayer::new(router, experts(num_experts, hidden, inter, &device), shared);
            if let Some(scale) = scale {
                layer = layer.with_shared_expert_scale(scale).unwrap();
            }

            let input = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device),
                false,
            );
            let out = layer.forward(&client, &input).unwrap();
            out.output.tensor().contiguous().unwrap().to_vec()
        }

        // Baseline: no shared expert at all, so only routed experts contribute.
        let routed_only = {
            let (client, device) = cpu_setup();
            let (hidden, inter, num_experts, top_k) = (4, 8, 2, 1);
            let gate_w =
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device);
            let router =
                MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), false);
            let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);
            let input = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device),
                false,
            );
            let out = layer.forward(&client, &input).unwrap();
            out.output.tensor().contiguous().unwrap().to_vec::<f32>()
        };

        let unscaled = run(None);
        let half = run(Some(0.5));

        for i in 0..unscaled.len() {
            // Default stays exactly as before this option existed.
            let shared_contribution = unscaled[i] - routed_only[i];
            assert!(
                shared_contribution.abs() > 1e-6,
                "test setup is degenerate: shared expert contributes nothing"
            );
            // Scaling by 0.5 must halve precisely that contribution.
            let expected = routed_only[i] + 0.5 * shared_contribution;
            assert!(
                (half[i] - expected).abs() < 1e-5,
                "index {i}: expected {expected}, got {}",
                half[i]
            );
        }
    }

    #[test]
    fn shared_expert_scale_rejects_non_finite() {
        let (_client, device) = cpu_setup();
        let gate_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device);
        let router = MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(2, 1), false);
        let layer = MoeLayer::new(router, experts(2, 4, 8, &device), None);
        assert!(layer.with_shared_expert_scale(f32::NAN).is_err());
    }

    #[test]
    fn z_loss_produces_gate_gradient() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let inter = 8;
        let num_experts = 3;
        let top_k = 2;

        // Asymmetric weights and inputs keep this from passing by accidental
        // symmetry if z_loss ever stops being connected to the gate.
        let gate_w = Tensor::<CpuRuntime>::from_slice(
            &[
                0.7f32, -0.2, 0.15, 0.4, -0.35, 0.6, 0.25, -0.1, 0.05, -0.45, 0.8, 0.3,
            ],
            &[num_experts, hidden],
            &device,
        );
        let router = MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), true);
        let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    0.3f32, -0.7, 1.1, 0.2, 0.8, 0.4, -0.3, 0.9, -0.6, 0.5, 0.7, -0.2,
                ],
                &[3, hidden],
                &device,
            ),
            false,
        );

        let result = layer.forward(&client, &input).unwrap();
        let grads = numr::autograd::backward(&result.z_loss, &client).unwrap();
        let gate_id = layer.router().gate().parameters()[0].0;
        let gate_grad = grads
            .get(gate_id)
            .expect("z_loss must produce a gradient for the gate weight")
            .contiguous()
            .unwrap();
        let magnitude: f32 = gate_grad.to_vec::<f32>().iter().map(|v| v.abs()).sum();
        assert!(
            magnitude > 1e-8,
            "gate gradient from z_loss is all zeros ({magnitude}) — graph is severed"
        );
    }
}

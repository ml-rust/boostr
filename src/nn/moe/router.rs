//! MoE Router — top-k expert gating with load balancing

use crate::error::{Error, Result};
use crate::nn::Linear;
use numr::autograd::{
    Var, var_add, var_div, var_div_scalar, var_gather, var_mean, var_mul, var_mul_scalar,
    var_softmax, var_sum,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Load-balancing auxiliary loss formulation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeLoadBalanceLossMode {
    /// Switch Transformer loss: `num_experts * sum(P_e * N_e)`.
    #[default]
    Switch,
    /// Switch loss plus differentiable probability regularization:
    /// `num_experts * (sum(P_e^2) + sum(P_e * N_e))`.
    SwitchPlusSquaredProb,
}

/// Router configuration
#[derive(Clone, Copy, Debug)]
pub struct MoeRouterConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to route each token to
    pub top_k: usize,
    /// Softmax temperature for router logits. Must be finite and positive.
    pub router_temperature: f32,
    /// Load-balancing auxiliary loss formulation.
    pub load_balance_loss_mode: MoeLoadBalanceLossMode,
}

impl MoeRouterConfig {
    /// Create a router config with the historical boostr behavior.
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            router_temperature: 1.0,
            load_balance_loss_mode: MoeLoadBalanceLossMode::Switch,
        }
    }
}

/// MoE Router with top-k gating.
///
/// Routes tokens to experts using a learned gate projection.
pub struct MoeRouter<R: Runtime> {
    gate: Linear<R>,
    config: MoeRouterConfig,
}

/// Router output: selected expert indices, weights, logits, and auxiliary loss
pub struct RouterOutput<R: Runtime> {
    /// Expert weights per token: `[batch * seq, top_k]`
    pub weights: Var<R>,
    /// Expert indices per token: `[batch * seq, top_k]` (I64 tensor)
    pub indices: Tensor<R>,
    /// Raw gate logits after router-temperature scaling, before softmax:
    /// `[batch * seq, num_experts]`.
    pub logits: Var<R>,
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
    /// Returns: RouterOutput with weights, indices, logits, and aux_loss
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
        if !self.config.router_temperature.is_finite() || self.config.router_temperature <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "router_temperature",
                reason: format!(
                    "expected finite positive value, got {}",
                    self.config.router_temperature
                ),
            });
        }

        // gate logits: [num_tokens, num_experts]
        let raw_logits = self.gate.forward(client, x)?;
        let logits = if self.config.router_temperature == 1.0 {
            raw_logits
        } else {
            var_div_scalar(&raw_logits, self.config.router_temperature as f64, client)
                .map_err(Error::Numr)?
        };

        // softmax over experts dim
        let probs = var_softmax(&logits, -1, client).map_err(Error::Numr)?;

        // top-k selection via numr's SortingOps (stays on device).
        // Only the INDICES are taken from the raw op: selection is discrete and
        // non-differentiable. The corresponding weights must be re-gathered from
        // the tracked `probs` Var, or the gate receives no gradient at all.
        let (_, top_indices) = client
            .topk(probs.tensor(), self.config.top_k, -1, true, true)
            .map_err(Error::Numr)?;

        let top_values = var_gather(&probs, 1, &top_indices, client).map_err(Error::Numr)?;

        // Normalize top-k weights to sum to 1 (tracked)
        let weight_sum = var_sum(&top_values, &[1], true, client).map_err(Error::Numr)?;
        let normalized_weights = var_div(&top_values, &weight_sum, client).map_err(Error::Numr)?;

        // Compute load balancing auxiliary loss
        let aux_loss = self.compute_aux_loss(client, &probs, &top_indices)?;

        Ok(RouterOutput {
            weights: normalized_weights,
            indices: top_indices,
            logits,
            aux_loss,
        })
    }

    /// Compute load balancing auxiliary loss.
    ///
    /// Switch loss = num_experts * sum(P_e * N_e) where:
    /// - P_e = mean probability assigned to expert e across all tokens
    /// - N_e = fraction of tokens routed to expert e
    ///
    /// `SwitchPlusSquaredProb` adds `num_experts * sum(P_e^2)`.
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
        let num_tokens = probs.tensor().shape()[0];
        let num_experts = self.config.num_experts;
        let k = self.config.top_k;

        // P_e: mean probability per expert [num_experts].
        // MUST be a tracked reduction over `probs` — reducing the raw tensor and
        // re-wrapping with Var::new severs the graph and leaves the gate untrained.
        let p_e_var = var_mean(probs, &[0], false, client).map_err(Error::Numr)?;

        // N_e: fraction of tokens routed to each expert (on-device via bincount)
        let flat_indices = indices.reshape(&[indices.numel()]).map_err(Error::Numr)?;
        let counts = client
            .bincount(&flat_indices, None, num_experts)
            .map_err(Error::Numr)?;
        let counts_f32 = client.cast(&counts, DType::F32).map_err(Error::Numr)?;
        let total = (num_tokens * k) as f64;
        let n_e = client.div_scalar(&counts_f32, total)?;

        // aux_loss = num_experts * sum(P_e * N_e).
        // N_e comes from discrete token counts and is correctly a constant leaf;
        // the gradient path runs through P_e.
        let n_e_var = Var::new(n_e, false);

        let pn = var_mul(&p_e_var, &n_e_var, client).map_err(Error::Numr)?;
        let switch_sum = var_sum(&pn, &[0], false, client).map_err(Error::Numr)?;
        let loss_sum = match self.config.load_balance_loss_mode {
            MoeLoadBalanceLossMode::Switch => switch_sum,
            MoeLoadBalanceLossMode::SwitchPlusSquaredProb => {
                // The count term is zero for unrouted experts. P_e^2 stays
                // differentiable for every expert because it is computed from
                // the tracked dense softmax probabilities.
                let p_squared = var_mul(&p_e_var, &p_e_var, client).map_err(Error::Numr)?;
                let p_squared_sum =
                    var_sum(&p_squared, &[0], false, client).map_err(Error::Numr)?;
                var_add(&p_squared_sum, &switch_sum, client).map_err(Error::Numr)?
            }
        };
        let loss = var_mul_scalar(&loss_sum, num_experts as f64, client).map_err(Error::Numr)?;

        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::SortingOps;
    use numr::runtime::cpu::CpuRuntime;

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        );
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tol,
                "{label}[{idx}]={actual}, expected={expected}, tol={tol}"
            );
        }
    }

    fn var_values(var: &Var<CpuRuntime>) -> Vec<f32> {
        var.tensor().contiguous().unwrap().to_vec::<f32>()
    }

    fn gate_gradient_values(
        grads: &numr::autograd::GradStore<CpuRuntime>,
        router: &MoeRouter<CpuRuntime>,
    ) -> Vec<f32> {
        let gate_id = router.gate().parameters()[0].0;
        let grad = grads
            .get(gate_id)
            .expect("loss must produce a gradient for the gate weight");
        grad.contiguous().unwrap().to_vec::<f32>()
    }

    #[test]
    fn test_router_output_shapes() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let num_experts = 4;
        let top_k = 2;

        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 16], &[num_experts, hidden], &device)
                .unwrap();

        let config = MoeRouterConfig::new(num_experts, top_k);
        let router = MoeRouter::from_tensor(gate_w, config, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device).unwrap(),
            false,
        );
        let output = router.route(&client, &input).unwrap();

        assert_eq!(output.weights.shape(), &[3, top_k]);
        assert_eq!(output.indices.shape(), &[3, top_k]);
        assert_eq!(output.logits.shape(), &[3, num_experts]);
        assert_eq!(output.aux_loss.tensor().numel(), 1);
    }

    /// Build a router whose gate is trainable, plus a batch of distinct inputs.
    fn trainable_router_with_config(
        config: MoeRouterConfig,
    ) -> (
        <CpuRuntime as Runtime>::Client,
        MoeRouter<CpuRuntime>,
        Var<CpuRuntime>,
    ) {
        let (client, device) = cpu_setup();
        let hidden = 4;
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.top_k, 2);

        // Asymmetric gate + inputs: a symmetric setup can produce a genuinely
        // zero gradient and would pass even with the graph severed.
        let gate_w = Tensor::<CpuRuntime>::from_slice(
            &[
                0.9f32, -0.2, 0.4, 0.1, -0.5, 0.7, 0.05, 0.3, 0.2, 0.15, -0.8, 0.6, -0.1, 0.35,
                0.25, -0.45,
            ],
            &[config.num_experts, hidden],
            &device,
        )
        .unwrap();
        let router = MoeRouter::from_tensor(gate_w, config, true);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    0.3f32, -0.7, 1.1, 0.2, 0.8, 0.4, -0.3, 0.9, -0.6, 0.5, 0.7, -0.2,
                ],
                &[3, hidden],
                &device,
            )
            .unwrap(),
            false,
        );
        (client, router, input)
    }

    fn trainable_router() -> (
        <CpuRuntime as Runtime>::Client,
        MoeRouter<CpuRuntime>,
        Var<CpuRuntime>,
    ) {
        trainable_router_with_config(MoeRouterConfig::new(4, 2))
    }

    /// The load-balancing aux loss must reach the gate weight in every mode.
    ///
    /// Regression: `p_e` was reduced from `probs.tensor()` (a raw op) and
    /// re-wrapped with `Var::new`, which creates a LEAF with no grad_fn. The
    /// aux loss then backpropagated into an orphan id and the gate never
    /// received a gradient — load balancing silently did nothing.
    #[test]
    fn aux_loss_modes_produce_gate_gradient() {
        for mode in [
            MoeLoadBalanceLossMode::Switch,
            MoeLoadBalanceLossMode::SwitchPlusSquaredProb,
        ] {
            let config = MoeRouterConfig {
                load_balance_loss_mode: mode,
                ..MoeRouterConfig::new(4, 2)
            };
            let (client, router, input) = trainable_router_with_config(config);
            let out = router.route(&client, &input).unwrap();

            let grads = numr::autograd::backward(&out.aux_loss, &client).unwrap();
            let magnitude: f32 = gate_gradient_values(&grads, &router)
                .iter()
                .map(|v| v.abs())
                .sum();
            assert!(
                magnitude > 1e-8,
                "gate gradient from {mode:?} aux_loss is all zeros ({magnitude}) — graph is severed"
            );
        }
    }

    #[test]
    fn router_temperature_one_matches_unscaled_route_and_temperature_changes_probs() {
        let (client, router, input) = trainable_router();
        let out = router.route(&client, &input).unwrap();

        // Temperature 1.0 is the historical path: raw gate logits go directly
        // into softmax/top-k/aux loss with no numerical change.
        let raw_logits = router.gate().forward(&client, &input).unwrap();
        assert_close(
            &var_values(&out.logits),
            &var_values(&raw_logits),
            1e-7,
            "temperature=1 logits",
        );

        let raw_probs = var_softmax(&raw_logits, -1, &client).unwrap();
        let (_, raw_indices) = client
            .topk(raw_probs.tensor(), router.config().top_k, -1, true, true)
            .unwrap();
        let raw_top_values = var_gather(&raw_probs, 1, &raw_indices, &client).unwrap();
        let raw_weight_sum = var_sum(&raw_top_values, &[1], true, &client).unwrap();
        let raw_weights = var_div(&raw_top_values, &raw_weight_sum, &client).unwrap();
        let raw_aux_loss = router
            .compute_aux_loss(&client, &raw_probs, &raw_indices)
            .unwrap();

        assert_eq!(out.indices.to_vec::<i64>(), raw_indices.to_vec::<i64>());
        assert_close(
            &var_values(&out.weights),
            &var_values(&raw_weights),
            1e-6,
            "temperature=1 weights",
        );
        assert_close(
            &var_values(&out.aux_loss),
            &var_values(&raw_aux_loss),
            1e-6,
            "temperature=1 aux_loss",
        );

        let hot_config = MoeRouterConfig {
            router_temperature: 0.5,
            ..MoeRouterConfig::new(4, 2)
        };
        let hot_router =
            MoeRouter::from_tensor(router.gate().weight().tensor().clone(), hot_config, true);
        let hot_out = hot_router.route(&client, &input).unwrap();
        let probs_one = var_softmax(&out.logits, -1, &client).unwrap();
        let probs_hot = var_softmax(&hot_out.logits, -1, &client).unwrap();
        let probs_one_values = var_values(&probs_one);
        let probs_hot_values = var_values(&probs_hot);
        let prob_delta: f32 = probs_one_values
            .iter()
            .zip(&probs_hot_values)
            .map(|(a, b)| (*a - *b).abs())
            .sum();
        assert!(
            prob_delta > 1e-4,
            "changing router_temperature must change the softmax distribution"
        );
    }

    #[test]
    fn router_temperature_must_be_positive_and_finite() {
        let (client, device) = cpu_setup();
        let hidden = 2;
        let num_experts = 2;
        let gate_w = Tensor::<CpuRuntime>::from_slice(
            &[0.2f32, -0.1, -0.3, 0.4],
            &[num_experts, hidden],
            &device,
        )
        .unwrap();
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.25], &[1, hidden], &device).unwrap(),
            false,
        );

        for bad_temperature in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let config = MoeRouterConfig {
                router_temperature: bad_temperature,
                ..MoeRouterConfig::new(num_experts, 1)
            };
            let router = MoeRouter::from_tensor(gate_w.clone(), config, false);
            match router.route(&client, &input) {
                Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, "router_temperature"),
                Err(err) => panic!("expected router_temperature error, got {err:?}"),
                Ok(_) => panic!("router_temperature={bad_temperature} should be rejected"),
            }
        }
    }

    /// SwitchPlusSquaredProb gives a zero-token ("dead") expert differentiable
    /// pressure that does not depend on it being routed to.
    ///
    /// Setup routes every token to experts 0 and 1, never expert 2. Under the
    /// Switch term, expert 2's own count N_2 is 0, so the only gradient it can
    /// receive is indirect softmax coupling through the experts that DID win.
    /// The squared-prob term adds a contribution that exists regardless of
    /// routing counts.
    ///
    /// The assertion is on the DIFFERENCE between the two modes rather than on
    /// the Switch gradient being ~0. Switch's coupling term is not identically
    /// zero, and an input tuned to cancel it exactly would make this test brittle
    /// against any change to the backward path — it would then fail for reasons
    /// unrelated to the property under test.
    #[test]
    fn squared_prob_mode_gives_dead_expert_extra_gradient() {
        fn run_mode(mode: MoeLoadBalanceLossMode) -> (RouterOutput<CpuRuntime>, Vec<f32>) {
            let (client, device) = cpu_setup();
            let gate_w =
                Tensor::<CpuRuntime>::from_slice(&[2.0f32, -1.0, 0.0], &[3, 1], &device).unwrap();
            let config = MoeRouterConfig {
                load_balance_loss_mode: mode,
                ..MoeRouterConfig::new(3, 1)
            };
            let router = MoeRouter::from_tensor(gate_w, config, true);
            let input = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[1.0f32, -0.5], &[2, 1], &device).unwrap(),
                false,
            );
            let out = router.route(&client, &input).unwrap();
            let grads = numr::autograd::backward(&out.aux_loss, &client).unwrap();
            let gate_grad = gate_gradient_values(&grads, &router);
            (out, gate_grad)
        }

        let (switch_out, switch_grad) = run_mode(MoeLoadBalanceLossMode::Switch);
        let (plus_out, plus_grad) = run_mode(MoeLoadBalanceLossMode::SwitchPlusSquaredProb);

        // Expert 2 is never selected by either mode: it is genuinely dead.
        let switch_indices = switch_out.indices.contiguous().unwrap().to_vec::<i64>();
        let plus_indices = plus_out.indices.contiguous().unwrap().to_vec::<i64>();
        assert_eq!(switch_indices, vec![0, 1]);
        assert_eq!(plus_indices, vec![0, 1]);

        let dead_expert = 2;
        let extra = plus_grad[dead_expert] - switch_grad[dead_expert];
        assert!(
            extra.abs() > 1e-3,
            "squared-prob term must add gradient at a zero-token expert: \
             switch={}, plus={}, difference={extra}",
            switch_grad[dead_expert],
            plus_grad[dead_expert]
        );
        assert!(
            plus_grad[dead_expert].abs() > 1e-3,
            "dead expert must end up with usable gradient, got {}",
            plus_grad[dead_expert]
        );
    }

    /// The routing weights must also reach the gate.
    ///
    /// Regression: top-k weights were taken straight from `client.topk` on the
    /// raw tensor and wrapped with `Var::new`, so no gradient flowed to the gate
    /// through the expert-combination path either.
    #[test]
    fn routing_weights_produce_gate_gradient() {
        let (client, router, input) = trainable_router();
        let out = router.route(&client, &input).unwrap();

        // Reduce the weights alone, so only the weight path can supply gradient.
        let scalar = var_sum(&out.weights, &[0, 1], false, &client).unwrap();
        let grads = numr::autograd::backward(&scalar, &client).unwrap();
        let gate_id = router.gate().parameters()[0].0;

        assert!(
            grads.get(gate_id).is_some(),
            "routing weights must carry gradient back to the gate weight"
        );
    }
}

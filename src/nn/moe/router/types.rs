//! Router configuration and output types. The algorithm lives in `super::route`.

use numr::autograd::Var;
use numr::runtime::Runtime;
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

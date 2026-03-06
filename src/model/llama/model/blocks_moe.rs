//! MoE MLP block for Llama-family MoE models (Mixtral, Qwen2-MoE, DeepSeek).
//!
//! Replaces `LlamaMlp` in transformer blocks where MoE routing is used.
//! Uses `MoEOps` trait operations (top-k routing, permute, grouped GEMM, unpermute)
//! which have dedicated kernels on CPU, CUDA, and WebGPU.

use crate::error::{Error, Result};
use crate::model::config::moe::MoeConfig;
use crate::model::traits::ModelClient;
use crate::nn::MaybeQuantLinear;
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::{Arc, RwLock};

/// Weight tensors for a single MoE expert (gate, up, down projections).
///
/// Each tensor is a 2-D slice extracted from the layer's stacked weight tensors:
/// - `gate_proj`: `[hidden_size, intermediate_size]`
/// - `up_proj`:   `[hidden_size, intermediate_size]`
/// - `down_proj`: `[intermediate_size, hidden_size]`
pub struct ExpertWeights<R: Runtime> {
    pub gate_proj: Tensor<R>,
    pub up_proj: Tensor<R>,
    pub down_proj: Tensor<R>,
}

/// MoE MLP layer for inference.
///
/// Each expert is a SwiGLU MLP (gate_proj, up_proj, down_proj).
/// Expert weights are stored as stacked tensors: `[num_experts, dim_in, dim_out]`
/// for efficient grouped GEMM.
///
/// The stacked tensors are wrapped in `Arc<RwLock<_>>` to allow in-place expert
/// weight swapping at runtime (CPU↔GPU transfers) without requiring `&mut self`.
pub struct LlamaMoeMlp<R: Runtime> {
    /// Gate projection weights: `[num_experts, hidden_size, intermediate_size]`
    gate_weights: Arc<RwLock<Tensor<R>>>,
    /// Up projection weights: `[num_experts, hidden_size, intermediate_size]`
    up_weights: Arc<RwLock<Tensor<R>>>,
    /// Down projection weights: `[num_experts, intermediate_size, hidden_size]`
    down_weights: Arc<RwLock<Tensor<R>>>,
    /// Router gate linear: projects hidden_size → num_experts
    router_gate: MaybeQuantLinear<R>,
    /// Optional shared expert
    shared_gate_proj: Option<MaybeQuantLinear<R>>,
    shared_up_proj: Option<MaybeQuantLinear<R>>,
    shared_down_proj: Option<MaybeQuantLinear<R>>,
    /// MoE configuration
    config: MoeConfig,
}

impl<R: Runtime<DType = DType>> LlamaMoeMlp<R> {
    pub fn new(
        gate_weights: Tensor<R>,
        up_weights: Tensor<R>,
        down_weights: Tensor<R>,
        router_gate: MaybeQuantLinear<R>,
        config: MoeConfig,
    ) -> Self {
        Self {
            gate_weights: Arc::new(RwLock::new(gate_weights)),
            up_weights: Arc::new(RwLock::new(up_weights)),
            down_weights: Arc::new(RwLock::new(down_weights)),
            router_gate,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            config,
        }
    }

    /// Extract the weight tensors for a single expert as a 2-D view.
    ///
    /// The returned tensors are zero-copy `narrow` views into the stacked tensors.
    /// Returns `None` if `expert_id >= num_experts`.
    pub fn get_expert_weights(&self, expert_id: usize) -> Option<ExpertWeights<R>> {
        let num_experts = self.config.num_experts;
        if expert_id >= num_experts {
            return None;
        }
        // Acquire read locks; propagate poisoned-lock as None.
        let gate = self.gate_weights.read().ok()?;
        let up = self.up_weights.read().ok()?;
        let down = self.down_weights.read().ok()?;

        // narrow(dim=0, start=expert_id, length=1) → [1, in, out]
        // squeeze dim 0 → [in, out] (contiguous view)
        let gate_slice = gate.narrow(0, expert_id, 1).ok()?.contiguous();
        let up_slice = up.narrow(0, expert_id, 1).ok()?.contiguous();
        let down_slice = down.narrow(0, expert_id, 1).ok()?.contiguous();

        // Determine 2-D shapes by dropping the leading expert dim.
        let gate_shape = {
            let s = gate_slice.shape();
            vec![s[1], s[2]]
        };
        let up_shape = {
            let s = up_slice.shape();
            vec![s[1], s[2]]
        };
        let down_shape = {
            let s = down_slice.shape();
            vec![s[1], s[2]]
        };

        // Reshape [1, a, b] → [a, b]
        let gate_2d = gate_slice.reshape(&gate_shape).ok()?;
        let up_2d = up_slice.reshape(&up_shape).ok()?;
        let down_2d = down_slice.reshape(&down_shape).ok()?;

        Some(ExpertWeights {
            gate_proj: gate_2d,
            up_proj: up_2d,
            down_proj: down_2d,
        })
    }

    /// Replace the weight tensors for a single expert in-place.
    ///
    /// `weights.gate_proj` must be `[hidden_size, intermediate_size]`,
    /// `weights.up_proj` must be `[hidden_size, intermediate_size]`,
    /// `weights.down_proj` must be `[intermediate_size, hidden_size]`.
    ///
    /// Internally reconstructs the full stacked tensor by concatenating the
    /// before-slice, the new expert weights, and the after-slice along dim 0.
    ///
    /// Returns an error if the lock is poisoned, if `expert_id` is out of range,
    /// or if the `cat` operation fails.
    pub fn set_expert_weights(&self, expert_id: usize, weights: ExpertWeights<R>) -> Result<()>
    where
        R::Client: ShapeOps<R>,
    {
        let num_experts = self.config.num_experts;
        if expert_id >= num_experts {
            return Err(Error::ModelError {
                reason: format!(
                    "expert_id {} out of range (num_experts={})",
                    expert_id, num_experts
                ),
            });
        }

        // Helper: rebuild a stacked [num_experts, a, b] tensor by replacing slice at expert_id.
        // new_slice is 2-D [a, b]; we unsqueeze to [1, a, b] then cat with neighbours.
        fn rebuild_stacked<R: Runtime<DType = DType>>(
            stacked: &Tensor<R>,
            expert_id: usize,
            new_slice_2d: Tensor<R>,
        ) -> Result<Tensor<R>>
        where
            R::Client: ShapeOps<R>,
        {
            let num_experts = stacked.shape()[0];
            let client = R::default_client(stacked.device());

            // Unsqueeze new slice: [a, b] → [1, a, b]
            let new_shape = {
                let s = new_slice_2d.shape();
                vec![1usize, s[0], s[1]]
            };
            let new_1d = new_slice_2d
                .reshape(&new_shape)
                .map_err(|e| Error::ModelError {
                    reason: e.to_string(),
                })?;

            // Build list of slices to cat
            let mut parts: Vec<Tensor<R>> = Vec::with_capacity(3);

            if expert_id > 0 {
                let before = stacked
                    .narrow(0, 0, expert_id)
                    .map_err(|e| Error::ModelError {
                        reason: e.to_string(),
                    })?
                    .contiguous();
                parts.push(before);
            }

            parts.push(new_1d.contiguous());

            let after_start = expert_id + 1;
            if after_start < num_experts {
                let after = stacked
                    .narrow(0, after_start, num_experts - after_start)
                    .map_err(|e| Error::ModelError {
                        reason: e.to_string(),
                    })?
                    .contiguous();
                parts.push(after);
            }

            let refs: Vec<&Tensor<R>> = parts.iter().collect();
            client.cat(&refs, 0).map_err(|e| Error::ModelError {
                reason: e.to_string(),
            })
        }

        // Gate weights
        {
            let read_guard = self.gate_weights.read().map_err(|_| Error::ModelError {
                reason: "gate_weights lock poisoned".into(),
            })?;
            let new_stacked = rebuild_stacked(&*read_guard, expert_id, weights.gate_proj)?;
            drop(read_guard);
            let mut write_guard = self.gate_weights.write().map_err(|_| Error::ModelError {
                reason: "gate_weights lock poisoned".into(),
            })?;
            *write_guard = new_stacked;
        }

        // Up weights
        {
            let read_guard = self.up_weights.read().map_err(|_| Error::ModelError {
                reason: "up_weights lock poisoned".into(),
            })?;
            let new_stacked = rebuild_stacked(&*read_guard, expert_id, weights.up_proj)?;
            drop(read_guard);
            let mut write_guard = self.up_weights.write().map_err(|_| Error::ModelError {
                reason: "up_weights lock poisoned".into(),
            })?;
            *write_guard = new_stacked;
        }

        // Down weights
        {
            let read_guard = self.down_weights.read().map_err(|_| Error::ModelError {
                reason: "down_weights lock poisoned".into(),
            })?;
            let new_stacked = rebuild_stacked(&*read_guard, expert_id, weights.down_proj)?;
            drop(read_guard);
            let mut write_guard = self.down_weights.write().map_err(|_| Error::ModelError {
                reason: "down_weights lock poisoned".into(),
            })?;
            *write_guard = new_stacked;
        }

        Ok(())
    }

    pub fn with_shared_expert(
        mut self,
        gate_proj: MaybeQuantLinear<R>,
        up_proj: MaybeQuantLinear<R>,
        down_proj: MaybeQuantLinear<R>,
    ) -> Self {
        self.shared_gate_proj = Some(gate_proj);
        self.shared_up_proj = Some(up_proj);
        self.shared_down_proj = Some(down_proj);
        self
    }

    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// MoE forward: route → permute → grouped SwiGLU GEMM → unpermute.
    ///
    /// Input: `[num_tokens, hidden_size]` (already flattened by caller)
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + MoEOps<R>,
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
        let input = x.tensor();
        let num_tokens = input.shape()[0];

        // Acquire read locks on expert weight tensors for the duration of this forward pass.
        let gate_weights = self.gate_weights.read().map_err(|_| Error::ModelError {
            reason: "gate_weights lock poisoned".into(),
        })?;
        let up_weights = self.up_weights.read().map_err(|_| Error::ModelError {
            reason: "up_weights lock poisoned".into(),
        })?;
        let down_weights = self.down_weights.read().map_err(|_| Error::ModelError {
            reason: "down_weights lock poisoned".into(),
        })?;

        // 1. Router: gate logits → top-k expert selection
        let gate_logits_var = self.router_gate.forward(client, x)?;
        let (indices, weights) =
            client.moe_top_k_routing(gate_logits_var.tensor(), self.config.experts_per_tok)?;

        // 2. Permute tokens into expert-contiguous order
        let (permuted, offsets, sort_indices) =
            client.moe_permute_tokens(input, &indices, self.config.num_experts)?;

        // 3. SwiGLU grouped GEMM:
        //    gate_out = silu(permuted @ gate_weights)
        //    up_out   = permuted @ up_weights
        //    hidden   = gate_out * up_out
        //    output   = hidden @ down_weights
        let gate_out = client.moe_grouped_gemm_fused(
            &permuted,
            &*gate_weights,
            &offsets,
            MoEActivation::SiLU,
        )?;
        let up_out = client.moe_grouped_gemm(&permuted, &*up_weights, &offsets)?;
        let hidden = client.mul(&gate_out, &up_out).map_err(Error::Numr)?;
        let expert_output = client.moe_grouped_gemm(&hidden, &*down_weights, &offsets)?;

        // 4. Unpermute back to original order with weighted combination
        let mut output =
            client.moe_unpermute_tokens(&expert_output, &sort_indices, &weights, num_tokens)?;

        // 5. Shared expert (always active)
        if let (Some(sg), Some(su), Some(sd)) = (
            &self.shared_gate_proj,
            &self.shared_up_proj,
            &self.shared_down_proj,
        ) {
            let sg_out = sg.forward(client, x)?;
            let su_out = su.forward(client, x)?;
            let sg_silu = numr::autograd::var_silu(&sg_out, client).map_err(Error::Numr)?;
            let sh = numr::autograd::var_mul(&sg_silu, &su_out, client).map_err(Error::Numr)?;
            let shared_out = sd.forward(client, &sh)?;
            output = client
                .add(&output, shared_out.tensor())
                .map_err(Error::Numr)?;
        }

        Ok(Var::new(output, false))
    }
}

/// Enum to represent either a dense MLP or an MoE MLP in a transformer block.
pub enum LlamaFfn<R: Runtime> {
    Dense(super::mlp::LlamaMlp<R>),
    Moe(LlamaMoeMlp<R>),
}

impl<R: Runtime<DType = DType>> LlamaFfn<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + MoEOps<R>,
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
        match self {
            LlamaFfn::Dense(mlp) => mlp.forward(client, x),
            LlamaFfn::Moe(moe) => moe.forward(client, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_moe_mlp_forward_shape() {
        use numr::ops::RandomOps;

        let (client, _device) = cpu_setup();
        let hidden = 8;
        let inter = 16;
        let num_experts = 4;
        let top_k = 2;

        let gate_w = client
            .randn(&[num_experts, hidden, inter], DType::F32)
            .unwrap();
        let up_w = client
            .randn(&[num_experts, hidden, inter], DType::F32)
            .unwrap();
        let down_w = client
            .randn(&[num_experts, inter, hidden], DType::F32)
            .unwrap();

        let router = MaybeQuantLinear::Standard(Linear::new(
            client.randn(&[num_experts, hidden], DType::F32).unwrap(),
            None,
            false,
        ));

        let config = MoeConfig {
            num_experts,
            experts_per_tok: top_k,
            shared_expert: None,
            intermediate_size: Some(inter),
            load_balance_alpha: 0.01,
            z_loss_alpha: 1e-3,
        };

        let moe_mlp = LlamaMoeMlp::new(gate_w, up_w, down_w, router, config);

        let input = Var::new(client.randn(&[3, hidden], DType::F32).unwrap(), false);
        let out = moe_mlp.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[3, hidden]);
    }
}

//! Parameter access and the trainable switch of a [`Qwen35AttentionBlock`].
//!
//! Names follow the loader layout under `self_attn` (see
//! `format::gguf::name_map`): `q_proj`, `k_proj`, `v_proj`, `o_proj`,
//! `q_norm.weight` and `k_norm.weight`. An adapted projection adds
//! `base.*`, `lora_a` and `lora_b`, as [`LoraLinear`](crate::nn::LoraLinear)
//! does. The query and gate share `q_proj`, so the block has no separate
//! gate weight.
//!
//! # `q_proj` row order
//!
//! A checkpoint interleaves query and gate rows per head. The block holds
//! `q_proj` rows regrouped to `[query | gate]` (see `layer`), and so does
//! the `lora_b` of a `q_proj` adapter. Every other parameter has the
//! checkpoint layout. [`Qwen35AttentionBlock::q_rows_to_checkpoint`] maps
//! block-layout rows back before a write.

use super::layer::Qwen35AttentionBlock;
use crate::error::Result;
use crate::nn::linear::regroup_leading_axis;
use crate::nn::{Module, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// All parameters with their stable autograd ids: `q_proj`, `k_proj`,
    /// `v_proj`, `o_proj`, then the q and k norm weights.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.attn_q.parameters();
        params.extend(self.attn_k.parameters());
        params.extend(self.attn_v.parameters());
        params.extend(self.attn_output.parameters());
        params.extend(self.qk_norm.parameters());
        params
    }

    /// Parameters that require grad.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named parameters in loader-layout names, in the order of
    /// [`Self::parameters`]. See the module docs for the `q_proj` rows.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "q_proj", self.attn_q.named_parameters());
        extend_named(&mut params, "k_proj", self.attn_k.named_parameters());
        extend_named(&mut params, "v_proj", self.attn_v.named_parameters());
        extend_named(&mut params, "o_proj", self.attn_output.named_parameters());
        params.extend(Module::named_parameters(&self.qk_norm));
        params
    }

    /// Set `requires_grad` on the q and k norm weights. Every `TensorId`
    /// stays.
    ///
    /// The projections keep the `requires_grad` they were built with. A
    /// LoRA adapter trains when attached with `trainable = true`; see
    /// [`Self::attach_adapter_trainable`].
    pub fn with_trainable(mut self, trainable: bool) -> Self {
        self.qk_norm = self.qk_norm.with_trainable(trainable);
        self
    }

    /// `true` when the q and k norm weights require grad.
    pub fn is_trainable(&self) -> bool {
        self.qk_norm.q_norm().weight().requires_grad()
    }

    /// `rows`, a `q_proj` weight or `q_proj` `lora_b` in the block's
    /// `[query | gate]` row order, in the checkpoint's per-head
    /// interleaved order. Trailing axes are untouched.
    ///
    /// # Errors
    ///
    /// `QuantError` when the row count is not a multiple of
    /// `2 * num_heads`; numr errors from the copy propagate.
    pub fn q_rows_to_checkpoint(&self, rows: &Tensor<R>) -> Result<Tensor<R>> {
        regroup_leading_axis(rows, 2, self.cfg.num_heads)
    }
}

impl<R: Runtime<DType = DType>> Module<R> for Qwen35AttentionBlock<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Qwen35AttentionBlock::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        Qwen35AttentionBlock::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Qwen35AttentionBlock::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        Qwen35AttentionBlock::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Qwen35AttentionProjection;
    use super::super::train_fixture::{Fixture, cfg32};
    use super::*;
    use crate::nn::LoraFactors;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn names(params: Vec<(String, &Var<CpuRuntime>)>) -> Vec<String> {
        params.into_iter().map(|(name, _)| name).collect()
    }

    fn trainable_names(block: &Qwen35AttentionBlock<CpuRuntime>) -> Vec<String> {
        block
            .named_parameters()
            .into_iter()
            .filter(|(_, var)| var.requires_grad())
            .map(|(name, _)| name)
            .collect()
    }

    /// A frozen block names every weight and trains nothing. Names and ids
    /// line up with `parameters`.
    #[test]
    fn frozen_block_names_every_weight() {
        let (_, device) = cpu_setup();
        let block = Fixture::f32(&cfg32(), 0x3a7e_0001).frozen_block(&device);
        let named = block.named_parameters();
        let ids: Vec<TensorId> = named.iter().map(|(_, var)| var.id()).collect();
        let want: Vec<TensorId> = block.parameters().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, want);
        assert_eq!(
            names(named),
            [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "q_norm.weight",
                "k_norm.weight",
            ]
        );
        assert!(block.trainable_parameters().is_empty());
        assert!(!block.is_trainable());
        assert_eq!(Module::parameters(&block).len(), 6);
    }

    /// `with_trainable` trains the norms only and keeps every id.
    #[test]
    fn with_trainable_trains_norms_only() {
        let (_, device) = cpu_setup();
        let frozen = Fixture::f32(&cfg32(), 0x3a7e_0002).frozen_block(&device);
        let ids: Vec<TensorId> = frozen.parameters().into_iter().map(|(id, _)| id).collect();
        let block = frozen.with_trainable(true);
        assert!(block.is_trainable());
        assert_eq!(trainable_names(&block), ["q_norm.weight", "k_norm.weight"]);
        let after: Vec<TensorId> = block.parameters().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, after);
        assert!(
            block
                .with_trainable(false)
                .trainable_parameters()
                .is_empty()
        );
    }

    /// Trainable adapters train their factors only, under the adapter
    /// names.
    #[test]
    fn trainable_adapters_train_factors_only() {
        let (client, device) = cpu_setup();
        let cfg = cfg32();
        let mut block = Fixture::f32(&cfg, 0x3a7e_0003).frozen_block(&device);
        let factors = |out: usize, inp: usize| LoraFactors {
            lora_a: Tensor::<CpuRuntime>::zeros(&[2, inp], DType::F32, &device).unwrap(),
            lora_b: Tensor::<CpuRuntime>::zeros(&[out, 2], DType::F32, &device).unwrap(),
            scaling: 1.0,
        };
        let q = factors(cfg.q_gate_dim(), cfg.hidden_size);
        let o = factors(cfg.hidden_size, cfg.q_dim());
        block
            .attach_adapter_trainable(&client, Qwen35AttentionProjection::Q, q, DType::F32, true)
            .unwrap();
        block
            .attach_adapter_trainable(
                &client,
                Qwen35AttentionProjection::Output,
                o,
                DType::F32,
                true,
            )
            .unwrap();
        assert_eq!(
            trainable_names(&block),
            [
                "q_proj.lora_a",
                "q_proj.lora_b",
                "o_proj.lora_a",
                "o_proj.lora_b"
            ]
        );
        assert!(names(block.named_parameters()).contains(&"q_proj.base.weight".to_string()));
    }

    /// The block's `q_proj` rows map back to the stored, per-head
    /// interleaved weight bit for bit.
    #[test]
    fn q_rows_round_trip_to_checkpoint_layout() {
        let (_, device) = cpu_setup();
        let fixture = Fixture::f32(&cfg32(), 0x3a7e_0004);
        let block = fixture.frozen_block(&device);
        let held = block.attn_q.weight().unwrap().tensor().clone();
        let stored = block.q_rows_to_checkpoint(&held).unwrap();
        assert_eq!(stored.to_vec::<f32>(), fixture.stored_q_f32());
        assert_ne!(held.to_vec::<f32>(), fixture.stored_q_f32());
    }
}

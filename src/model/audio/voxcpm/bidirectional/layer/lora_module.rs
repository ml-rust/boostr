use super::BidirectionalLayer;
use crate::error::Result;
use crate::nn::{LoraTargets, Module, child_params, extend_named};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> BidirectionalLayer<R> {
    /// Delegate to `BidirectionalAttention::apply_lora` and
    /// `BidirectionalMlp::apply_lora`, summing their counts. `prefix` is the
    /// dotted path the owning `LocalEncoder`/`LocalDit` would pass to
    /// `extend_named` for this layer, extended here by `"self_attn"`/`"mlp"`
    /// exactly as `Module::named_parameters` extends it above. No
    /// zero-match check here: see `BidirectionalAttention::apply_lora`'s
    /// doc comment.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = self.self_attn.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "self_attn"),
        )?;
        adapted += self.mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "mlp"),
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix`, delegating to `BidirectionalAttention::lora_projection_names`
    /// and `BidirectionalMlp::lora_projection_names` at the SAME
    /// `"self_attn"`/`"mlp"`-joined prefixes [`Self::apply_lora`] passes
    /// them, so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = self
            .self_attn
            .lora_projection_names(&LoraTargets::join(prefix, "self_attn"));
        names.extend(
            self.mlp
                .lora_projection_names(&LoraTargets::join(prefix, "mlp")),
        );
        names
    }

    /// Delegate to `BidirectionalAttention::load_lora_parameters` and
    /// `BidirectionalMlp::load_lora_parameters`, summing their counts. No
    /// prefix needed — unlike [`Self::apply_lora`], lookup is by ID, not by
    /// dotted path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.self_attn.load_lora_parameters(params)?;
        written += self.mlp.load_lora_parameters(params)?;
        Ok(written)
    }

    /// Delegate to `BidirectionalAttention::set_lora_trainable` and
    /// `BidirectionalMlp::set_lora_trainable`, summing their counts. No
    /// `targets`/`prefix` needed — unlike [`Self::apply_lora`], this is a
    /// blanket toggle over whatever is already adapted.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        self.self_attn.set_lora_trainable(trainable) + self.mlp.set_lora_trainable(trainable)
    }
}

/// Names ARE the field names (`input_layernorm`, `self_attn.*`,
/// `post_attention_layernorm`, `mlp.*`) — this matches the shared
/// `{layer_prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::bidirectional::loader`]) exactly, so the
/// owning `LocalEncoder`/`LocalDit` need only prefix by `{layer_prefix}` (a
/// numeric layer index under `encoder.layers`/`decoder.layers`) to reach the
/// full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for BidirectionalLayer<R> {
    fn parameters(&self) -> Vec<&numr::autograd::Var<R>> {
        let mut params = child_params(&self.input_layernorm);
        params.extend(child_params(&self.self_attn));
        params.extend(child_params(&self.post_attention_layernorm));
        params.extend(child_params(&self.mlp));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &numr::autograd::Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "input_layernorm",
            self.input_layernorm.named_parameters(),
        );
        extend_named(&mut params, "self_attn", self.self_attn.named_parameters());
        extend_named(
            &mut params,
            "post_attention_layernorm",
            self.post_attention_layernorm.named_parameters(),
        );
        extend_named(&mut params, "mlp", self.mlp.named_parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
    use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
    use crate::model::audio::voxcpm::local_dit::tests::{HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, norm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// A block-quantized projection contributes NO `Var<R>` (block-quantized
    /// storage has no gradient — see `MaybeLoraLinear::parameters`), while
    /// dense parameters (the layer's `RmsNorm` weights) still appear.
    #[test]
    fn quantized_projections_contribute_nothing_dense_norms_still_appear() {
        use crate::nn::{MaybeLoraLinear, MaybeQuantLinear};
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::QuantizeOps;

        let (client, device) = cpu_setup();
        const DIM: usize = 32; // Q4_0 block_size, so a single block quantizes cleanly.

        let quantized_linear = |seed: f32| {
            let data: Vec<f32> = (0..DIM * DIM)
                .map(|i| (i as f32 * 0.01 + seed).sin())
                .collect();
            let w = Tensor::<CpuRuntime>::from_slice(&data, &[DIM, DIM], &device).unwrap();
            let qt = client.quantize(&w, QuantFormat::Q4_0).unwrap();
            let linear: MaybeLoraLinear<CpuRuntime> =
                MaybeQuantLinear::Quantized(crate::nn::QuantLinear::new(qt, None)).into();
            linear
        };

        let layer = BidirectionalLayer {
            input_layernorm: norm(&device),
            self_attn: BidirectionalAttention {
                q_proj: quantized_linear(1.0),
                k_proj: quantized_linear(2.0),
                v_proj: quantized_linear(3.0),
                o_proj: quantized_linear(4.0),
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
            },
            post_attention_layernorm: norm(&device),
            mlp: BidirectionalMlp {
                gate_proj: quantized_linear(5.0),
                up_proj: quantized_linear(6.0),
                down_proj: quantized_linear(7.0),
            },
        };

        let named = layer.named_parameters();
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();

        // Every quantized projection is absent...
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            assert!(
                !names.iter().any(|n| n.starts_with(proj)),
                "quantized projection {proj} must contribute no Var<R>, found in {names:?}"
            );
        }
        // ...while the dense norms still appear.
        assert!(names.contains(&"input_layernorm.weight"));
        assert!(names.contains(&"post_attention_layernorm.weight"));
        assert_eq!(
            named.len(),
            2,
            "only the two RmsNorm weights should survive: {names:?}"
        );
    }
}

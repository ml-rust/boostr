//! The pipeline-stage forwards of a [`LlamaStage`]: embed, a range of held
//! layers, and the head.

use super::model::LlamaStage;
use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::llama::model::lm_head::{embed_with, final_logits};
use crate::model::llama::model::parts::run_layers;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LlamaStage<R> {
    /// Token ids `[B, S]` as hidden states `[B, S, hidden]`.
    ///
    /// # Errors
    ///
    /// A stage that does not start at layer 0, and so holds no embedding.
    pub fn forward_embed<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: IndexingOps<R> + ScalarOps<R> + TensorOps<R>,
    {
        let embed = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "this pipeline stage holds {} and no embedding: only the stage that starts \
                 at layer 0 takes token ids",
                    self.span
                ),
            })?;
        embed_with(client, embed, &self.config, input_ids)
    }

    /// Runs layers `start_layer..end_layer` with KV cache. The indices are
    /// the model's, and the range must lie inside the held span. Cache slot
    /// `0` holds layer `start_layer`.
    ///
    /// # Errors
    ///
    /// An empty range, a range outside the span, a cache whose layer count
    /// differs from the range length, or a layer that fails.
    #[allow(
        clippy::too_many_arguments,
        reason = "client, both residual streams, the cache, the layer range and the position"
    )]
    pub fn forward_layers_range<C>(
        &self,
        client: &C,
        hidden: Var<R>,
        prev_mlp_out: Option<Var<R>>,
        kv_cache: &mut LayeredKvCache<R>,
        start_layer: usize,
        end_layer: usize,
        position: usize,
    ) -> Result<(Var<R>, Option<Var<R>>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        if start_layer >= end_layer || !self.span.contains(start_layer, end_layer) {
            return Err(Error::ModelError {
                reason: format!(
                    "layer range {start_layer}..{end_layer} is not a non-empty range inside \
                     the {} this pipeline stage holds",
                    self.span
                ),
            });
        }
        let first = self.span.start();
        run_layers(
            client,
            &self.layers[start_layer - first..end_layer - first],
            start_layer,
            &self.rope,
            hidden,
            prev_mlp_out,
            kv_cache,
            position,
        )
    }

    /// The final norm and LM head over the last layer's output: logits
    /// `[B, S, vocab]`.
    ///
    /// # Errors
    ///
    /// A stage that does not end at the last layer, and so holds no head.
    pub fn forward_head<C>(
        &self,
        client: &C,
        hidden: Var<R>,
        prev_mlp_out: Option<Var<R>>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let head = self.head.as_ref().ok_or_else(|| Error::ModelError {
            reason: format!(
                "this pipeline stage holds {} and no LM head: only the stage that ends at \
                 layer {} returns logits",
                self.span,
                self.span.num_layers()
            ),
        })?;
        final_logits(
            client,
            &head.norm,
            &head.lm_head,
            &self.config,
            hidden,
            prev_mlp_out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::span::LayerSpan;
    use super::*;
    use crate::model::LoadedModel;
    use crate::nn::VarBuilder;
    use crate::test_utils::{
        TINY_MAX_SEQ_LEN, TINY_VOCAB, cpu_setup, random_llama_checkpoint, span_map,
    };
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const LAYERS: usize = 3;

    fn cache(layers: usize, device: &CpuDevice) -> LayeredKvCache<CpuRuntime> {
        LayeredKvCache::new_positional(layers, 1, 1, 4, TINY_MAX_SEQ_LEN, 8, DType::F32, device)
            .unwrap()
    }

    fn last_row(logits: &Tensor<CpuRuntime>) -> Vec<f32> {
        let all: Vec<f32> = logits.contiguous().unwrap().to_vec();
        all[all.len() - TINY_VOCAB..].to_vec()
    }

    /// Three stages, each built from the tensors of its own span only,
    /// give the logits of the whole model. Each stage takes every tensor
    /// its map holds, and no map holds a tensor of another stage's layers.
    #[test]
    fn test_three_range_loaded_stages_give_the_full_model_logits() {
        for tied in [false, true] {
            let (client, device) = cpu_setup();
            let (map, config) = random_llama_checkpoint(&client, &device, LAYERS, tied);
            let mut full_map = span_map(&map, &config, &LayerSpan::new(0..LAYERS, LAYERS).unwrap());
            let mut vb = VarBuilder::new(&mut full_map, &device);
            let full = LoadedModel::load(&config, &mut vb).unwrap();

            let tokens = [3i64, 1, 4, 1, 5];
            let ids =
                Tensor::<CpuRuntime>::from_slice(&tokens, &[1, tokens.len()], &device).unwrap();
            let mut full_cache = cache(LAYERS, &device);
            let want = last_row(
                &full
                    .forward_with_kv_cache(&ids, &mut full_cache, 0)
                    .unwrap(),
            );

            let mut stages = Vec::new();
            for l in 0..LAYERS {
                let span = LayerSpan::new(l..l + 1, LAYERS).unwrap();
                let mut part = span_map(&map, &config, &span);
                for name in part.names() {
                    let other = (0..LAYERS)
                        .filter(|&o| o != l)
                        .any(|o| name.starts_with(&format!("model.layers.{o}.")));
                    assert!(!other, "stage {l} holds {name}");
                }
                let mut vb = VarBuilder::new(&mut part, &device);
                let stage = LoadedModel::load_layers(&config, &mut vb, span).unwrap();
                assert!(
                    part.is_empty(),
                    "stage {l} left {:?}",
                    part.names().collect::<Vec<_>>()
                );
                stages.push(stage);
            }

            let mut hidden = stages[0].forward_embed(&ids).unwrap();
            let mut mlp = None;
            for (l, stage) in stages.iter().enumerate() {
                let mut c = cache(1, &device);
                (hidden, mlp) = stage
                    .forward_layers_range(hidden, mlp, &mut c, l, l + 1, 0)
                    .unwrap();
            }
            let got = last_row(&stages[LAYERS - 1].forward_head(hidden, mlp).unwrap());
            let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * scale,
                    "tied {tied}, logit {i}: {g} vs {w}"
                );
            }
        }
    }

    /// Every full-model path refuses a stage with an error, and each stage
    /// forward refuses a part the stage does not hold.
    #[test]
    fn test_a_stage_refuses_what_it_does_not_hold() {
        let (client, device) = cpu_setup();
        let (map, config) = random_llama_checkpoint(&client, &device, LAYERS, false);
        let span = LayerSpan::new(1..2, LAYERS).unwrap();
        let mut part = span_map(&map, &config, &span);
        let mut vb = VarBuilder::new(&mut part, &device);
        let stage = LoadedModel::load_layers(&config, &mut vb, span).unwrap();
        assert_eq!(stage.layer_span(), Some(&span));
        assert_eq!(stage.num_layers(), LAYERS);

        let ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[1, 2], &device).unwrap();
        let full_err = stage
            .forward_with_kv_cache(&ids, &mut cache(LAYERS, &device), 0)
            .unwrap_err()
            .to_string();
        assert!(
            full_err.contains("forward_with_kv_cache runs the whole model"),
            "{full_err}"
        );
        assert!(full_err.contains("layers 1..2 of 3"), "{full_err}");
        assert!(stage.forward_hidden(&ids).is_err());
        assert!(stage.paged_attention_refusal().is_some());
        let embed_err = stage.forward_embed(&ids).unwrap_err().to_string();
        assert!(embed_err.contains("no embedding"), "{embed_err}");

        let hidden = Var::new(
            Tensor::<CpuRuntime>::zeros(&[1, 2, config.hidden_size], DType::F32, &device).unwrap(),
            false,
        );
        let outside = stage
            .forward_layers_range(hidden.clone(), None, &mut cache(2, &device), 0, 2, 0)
            .unwrap_err()
            .to_string();
        assert!(
            outside.contains("is not a non-empty range inside"),
            "{outside}"
        );
        let head_err = stage.forward_head(hidden, None).unwrap_err().to_string();
        assert!(head_err.contains("no LM head"), "{head_err}");
    }

    /// A stage built from a map that lacks a tensor its span reads fails
    /// and names the tensor.
    #[test]
    fn test_a_missing_tensor_of_the_span_fails_the_load() {
        let (client, device) = cpu_setup();
        let (map, config) = random_llama_checkpoint(&client, &device, LAYERS, false);
        let span = LayerSpan::new(2..3, LAYERS).unwrap();
        let mut part = span_map(&map, &config, &span);
        part.remove("lm_head.weight");
        let mut vb = VarBuilder::new(&mut part, &device);
        let err = LoadedModel::load_layers(&config, &mut vb, span)
            .unwrap_err()
            .to_string();
        assert!(err.contains("lm_head.weight"), "{err}");
    }
}

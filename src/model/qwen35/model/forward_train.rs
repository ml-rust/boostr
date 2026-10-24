//! [`Qwen35Model::forward_train`]: the tracked training forward over a
//! token batch, with no KV cache and no GDN state.
//!
//! Each layer runs `attn_norm`, the mixer's `forward_train`, the residual
//! add, `post_attention_norm`, the FFN and the second residual add
//! (`ffn_residual`), every step on the autograd graph. Rope positions
//! start at 0, as for a prefill on an empty cache. With
//! `checkpoint_layers`, each layer is one activation-checkpoint segment.
//!
//! A tied head runs through the embedding's own `Var` (see `params`), so
//! the lookup and the head add to one gradient.

use super::build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
use super::forward_layers::ffn_residual;
use super::params::{attention_layer_named, gdn_layer_named};
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{Linear, RoPE};
use numr::autograd::{Var, checkpoint_with_client};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Training forward: logits `[batch, seq, vocab_size]` for
    /// `input_ids` `[batch, seq]`, tracked from the embedding lookup to
    /// `lm_head`.
    ///
    /// The math equals [`Self::forward_qwen35`] at `rope_pos = 0` over an
    /// empty KV cache and a zero GDN state. Gradients reach every
    /// parameter that requires grad: see [`Self::trainable_parameters`].
    /// `checkpoint_layers = true` recomputes each layer during backward
    /// instead of holding its activations; values and gradients stay the
    /// same.
    ///
    /// # Errors
    ///
    /// - [`Error::ModelError`] when `input_ids` is not `[batch, seq]` or
    ///   `seq` exceeds the rope table.
    /// - [`Error::InvalidArgument`] when the embedding, an FFN projection,
    ///   `lm_head` or a mixer projection is Hadamard-rotated, or a
    ///   projection has a decomposed AWQ/GPTQ base. `arg` names it.
    /// - Block, numr and kernel errors propagate.
    pub fn forward_train<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        checkpoint_layers: bool,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + 'static,
        R::Client: ModelClient<R>,
    {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::ModelError {
                reason: format!("qwen35: input_ids must be [batch, seq], got {shape:?}"),
            });
        }
        self.check_train()?;
        let positions = self.text_positions(shape[1], 0, input_ids.device())?;
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        for block in &self.blocks {
            hidden = match block {
                Qwen35Block::Attention(layer) if checkpoint_layers => {
                    attention_layer_checkpointed(layer, client, &hidden, &self.rope, &positions)?
                }
                Qwen35Block::Attention(layer) => {
                    attention_layer_train(layer, client, &hidden, &self.rope, &positions)?
                }
                Qwen35Block::Gdn(layer) if checkpoint_layers => {
                    gdn_layer_checkpointed(layer, client, &hidden)?
                }
                Qwen35Block::Gdn(layer) => gdn_layer_train(layer, client, &hidden)?,
            };
        }
        let hidden = self.norm.forward(client, &hidden)?;
        match self.tied_head() {
            Some(table) => {
                let head = Linear::with_ids(
                    table.tensor().clone(),
                    table.id(),
                    None,
                    table.requires_grad(),
                );
                head.forward(client, &hidden)
            }
            None => self.lm_head.forward(client, &hidden),
        }
    }

    /// Refuse every module outside the mixers that the tracked forward
    /// cannot carry a gradient through. Each mixer checks its own
    /// projections before it computes.
    fn check_train(&self) -> Result<()> {
        if self.embed_tokens.is_rotated() {
            return Err(Error::InvalidArgument {
                arg: "qwen35.embed_tokens",
                reason: "qwen35.embed_tokens is Hadamard-rotated, and the inverse rotation \
                         detaches the table. Load the model without the Hadamard attach to \
                         train it"
                    .to_string(),
            });
        }
        for block in &self.blocks {
            let mlp = match block {
                Qwen35Block::Attention(layer) => &layer.mlp,
                Qwen35Block::Gdn(layer) => &layer.mlp,
            };
            mlp.gate().check_tracked("qwen35.mlp.gate_proj")?;
            mlp.up().check_tracked("qwen35.mlp.up_proj")?;
            mlp.down().check_tracked("qwen35.mlp.down_proj")?;
        }
        self.lm_head.check_tracked("qwen35.lm_head")
    }
}

/// One attention layer: `x + mixer(attn_norm(x))`, then the FFN residual.
fn attention_layer_train<R, C>(
    layer: &Qwen35AttentionLayer<R>,
    client: &C,
    x: &Var<R>,
    rope: &RoPE<R>,
    positions: &Tensor<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: ModelClient<R>,
{
    let normed = layer.attn_norm.forward(client, x)?;
    let mixed = layer
        .mixer
        .forward_train(client, &normed, rope, positions)?;
    ffn_residual(client, x, &mixed, &layer.post_attention_norm, &layer.mlp)
}

/// One GDN layer: `x + mixer(attn_norm(x))`, then the FFN residual.
fn gdn_layer_train<R, C>(layer: &Qwen35GdnLayer<R>, client: &C, x: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: ModelClient<R>,
{
    let normed = layer.attn_norm.forward(client, x)?;
    let mixed = layer.mixer.forward_train(client, &normed, None)?;
    ffn_residual(client, x, &mixed, &layer.post_attention_norm, &layer.mlp)
}

/// [`attention_layer_train`] as one checkpoint segment over an alias of
/// `layer`, which keeps every `TensorId`.
fn attention_layer_checkpointed<R, C>(
    layer: &Qwen35AttentionLayer<R>,
    client: &C,
    x: &Var<R>,
    rope: &RoPE<R>,
    positions: &Tensor<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + 'static,
    R::Client: ModelClient<R>,
{
    let owned = Qwen35AttentionLayer {
        attn_norm: layer.attn_norm.alias(),
        mixer: layer.mixer.alias(),
        post_attention_norm: layer.post_attention_norm.alias(),
        mlp: layer.mlp.alias(),
    };
    let (rope, positions) = (rope.alias(), positions.clone());
    let named = attention_layer_named(layer);
    segment(client, x, &named, move |input, c: &C| {
        attention_layer_train(&owned, c, input, &rope, &positions)
    })
}

/// [`gdn_layer_train`] as one checkpoint segment over an alias of `layer`.
fn gdn_layer_checkpointed<R, C>(layer: &Qwen35GdnLayer<R>, client: &C, x: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + 'static,
    R::Client: ModelClient<R>,
{
    let owned = Qwen35GdnLayer {
        attn_norm: layer.attn_norm.alias(),
        mixer: layer.mixer.alias(),
        post_attention_norm: layer.post_attention_norm.alias(),
        mlp: layer.mlp.alias(),
    };
    let named = gdn_layer_named(layer);
    segment(client, x, &named, move |input, c: &C| {
        gdn_layer_train(&owned, c, input)
    })
}

/// Run `f` on `x` as one checkpoint segment whose inputs are `x` and
/// every parameter in `params` that requires grad.
fn segment<R, C, F>(client: &C, x: &Var<R>, params: &[(String, &Var<R>)], f: F) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + 'static,
    R::Client: ModelClient<R>,
    F: Fn(&Var<R>, &C) -> Result<Var<R>> + Send + Sync + 'static,
{
    let mut inputs: Vec<&Var<R>> = vec![x];
    inputs.extend(params.iter().map(|p| p.1).filter(|var| var.requires_grad()));
    checkpoint_with_client(
        move |segment_inputs, client: &C| {
            let input = segment_inputs.first().ok_or_else(|| {
                numr::error::Error::Internal("checkpointed qwen35 layer got no input".to_string())
            })?;
            f(input, client).map_err(|e| {
                numr::error::Error::Backend(format!("checkpointed qwen35 layer forward: {e}"))
            })
        },
        &inputs,
        client,
    )
    .map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{
        H_KV, HD, Head, MAX_POS, SEQ, VOCAB, chunked_model, rotated_model, tiny_model,
    };
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::inference::{LayeredGdnState, LayeredKvCache};
    use crate::model::hybrid::{GdnProjection, Qwen35AttentionProjection};
    use crate::nn::{LoraFactors, MlpProjection, cross_entropy_loss};
    use crate::test_utils::{cpu_setup, max_abs_diff};
    use numr::autograd::{GradStore, backward};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::TensorId;

    const RANK: usize = 2;
    const LR: f64 = 0.05;

    type Model = Qwen35Model<CpuRuntime>;

    fn tokens(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (0..SEQ).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[1, SEQ], device).unwrap()
    }

    /// Next-token targets of [`tokens`].
    fn targets(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (1..=SEQ).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[1, SEQ], device).unwrap()
    }

    fn prefill(client: &CpuClient, device: &CpuDevice, model: &Model) -> Vec<f32> {
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: H_KV,
            initial_capacity: MAX_POS,
            max_seq_len: MAX_POS,
            head_dim: HD,
            dtype: DType::F32,
        };
        let mut kv =
            LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
                .unwrap();
        let mut gdn = LayeredGdnState::<CpuRuntime>::zeros(
            model.num_gdn_layers(),
            model.gdn_config(),
            1,
            DType::F32,
            device,
        )
        .unwrap();
        model
            .forward_qwen35(client, &tokens(device), &mut kv, &mut gdn, 0)
            .unwrap()
            .to_vec::<f32>()
    }

    /// Host values of the gradient for `id`. A gradient can be a strided
    /// view, so it is made contiguous before the copy.
    fn grad_values(grads: &GradStore<CpuRuntime>, id: TensorId) -> Vec<f32> {
        let grad = grads
            .get(id)
            .unwrap_or_else(|| panic!("{id:?}: no gradient"));
        grad.contiguous().unwrap().to_vec::<f32>()
    }

    #[test]
    fn forward_train_matches_inference_prefill() {
        let (client, device) = cpu_setup();
        let models = [
            tiny_model(&device, 0x3a7e_0110),
            chunked_model(&device, 0x3a7e_0111, 4, false, Head::Tied),
        ];
        for model in &models {
            let want = prefill(&client, &device, model);
            for checkpoint in [false, true] {
                let got = model
                    .forward_train(&client, &tokens(&device), checkpoint)
                    .unwrap();
                assert_eq!(got.shape(), &[1, SEQ, VOCAB]);
                assert!(
                    !got.requires_grad(),
                    "frozen weights give an untracked output"
                );
                let diff = max_abs_diff(&got.tensor().to_vec::<f32>(), &want);
                assert!(diff < 1e-5, "checkpoint={checkpoint}: logits diff {diff}");
            }
        }
    }

    #[test]
    fn refuses_rotated_model() {
        let (client, device) = cpu_setup();
        let model = rotated_model(&device, 0x3a7e_0112);
        match model.forward_train(&client, &tokens(&device), false) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert!(arg.starts_with("qwen35"), "{arg}");
                assert!(reason.contains("Hadamard-rotated"), "{reason}");
            }
            Err(other) => panic!("expected InvalidArgument, got {other}"),
            Ok(_) => panic!("a rotated model must be refused"),
        }
    }

    // ---- LoRA training over a frozen base ----

    /// `(out, in)` of every adapted projection, in model parameter order:
    /// per layer the mixer's (`out_proj`; or `k_proj`, `o_proj`), then
    /// `mlp.down_proj`.
    fn adapter_shapes(model: &Model) -> Vec<(usize, usize)> {
        let hidden = model.config().hidden_size;
        let inter = model.config().intermediate_size.unwrap();
        let attn = model.attention_config();
        let mut shapes = Vec::new();
        for block in &model.blocks {
            match block {
                Qwen35Block::Gdn(_) => shapes.push((hidden, model.gdn_config().value_dim())),
                Qwen35Block::Attention(_) => {
                    shapes.push((attn.kv_dim(), hidden));
                    shapes.push((hidden, attn.q_dim()));
                }
            }
            shapes.push((hidden, inter));
        }
        shapes
    }

    /// Host values of every factor, `lora_a` then `lora_b` per adapter.
    fn initial_factors(model: &Model, seed: u32) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    0.3 * ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
                })
                .collect()
        };
        adapter_shapes(model)
            .into_iter()
            .flat_map(|(out, inp)| [next(RANK * inp), next(out * RANK)])
            .collect()
    }

    /// `model` with trainable adapters holding `values`, attached in the
    /// order of [`adapter_shapes`].
    fn adapted(mut model: Model, values: &[Vec<f32>], device: &CpuDevice) -> Model {
        let (client, _) = cpu_setup();
        let shapes = adapter_shapes(&model);
        let t = |d: &[f32], s: &[usize]| Tensor::<CpuRuntime>::from_slice(d, s, device).unwrap();
        let mut factors = shapes
            .iter()
            .zip(values.chunks(2))
            .map(|(&(out, inp), ab)| LoraFactors {
                lora_a: t(&ab[0], &[RANK, inp]),
                lora_b: t(&ab[1], &[out, RANK]),
                scaling: 0.5,
            });
        let mut next = || factors.next().unwrap();
        let dt = DType::F32;
        for block in &mut model.blocks {
            let mlp = match block {
                Qwen35Block::Gdn(layer) => {
                    let p = GdnProjection::SsmOut;
                    let mixer = &mut layer.mixer;
                    mixer
                        .attach_adapter_trainable(&client, p, next(), dt, true)
                        .unwrap();
                    &mut layer.mlp
                }
                Qwen35Block::Attention(layer) => {
                    for p in [
                        Qwen35AttentionProjection::K,
                        Qwen35AttentionProjection::Output,
                    ] {
                        let mixer = &mut layer.mixer;
                        mixer
                            .attach_adapter_trainable(&client, p, next(), dt, true)
                            .unwrap();
                    }
                    &mut layer.mlp
                }
            };
            let p = MlpProjection::Down;
            mlp.attach_adapter_trainable(&client, p, next(), dt, true)
                .unwrap();
        }
        model
    }

    fn loss(
        client: &CpuClient,
        device: &CpuDevice,
        model: &Model,
        checkpoint: bool,
    ) -> Var<CpuRuntime> {
        let logits = model
            .forward_train(client, &tokens(device), checkpoint)
            .unwrap();
        cross_entropy_loss(client, &logits, &targets(device)).unwrap()
    }

    #[test]
    fn frozen_base_trains_only_lora_factors() {
        let (client, device) = cpu_setup();
        let base = tiny_model(&device, 0x3a7e_0120);
        let values = initial_factors(&base, 0x3a7e_0121);
        let model = adapted(base, &values, &device);
        let grads = backward(&loss(&client, &device, &model, false), &client).unwrap();

        let trainable = model.trainable_parameters();
        assert_eq!(trainable.len(), values.len());
        for (name, var) in model.named_parameters() {
            let is_factor = name.ends_with("lora_a") || name.ends_with("lora_b");
            assert_eq!(var.requires_grad(), is_factor, "{name}");
            if is_factor {
                assert!(grads.get(var.id()).is_some(), "{name}: no gradient");
            } else if name.contains("proj") || name.starts_with("lm_head") {
                assert!(
                    grads.get(var.id()).is_none(),
                    "frozen {name} gets a gradient"
                );
            }
        }
    }

    #[test]
    fn checkpointed_gradients_equal_plain_gradients() {
        let (client, device) = cpu_setup();
        let base = tiny_model(&device, 0x3a7e_0122);
        let values = initial_factors(&base, 0x3a7e_0123);
        let model = adapted(base, &values, &device);
        let plain = backward(&loss(&client, &device, &model, false), &client).unwrap();
        let ckpt = backward(&loss(&client, &device, &model, true), &client).unwrap();
        let trainable = model.trainable_parameters();
        let ids: Vec<TensorId> = trainable.into_iter().map(|(id, _)| id).collect();
        for id in ids {
            let a = grad_values(&plain, id);
            let b = grad_values(&ckpt, id);
            assert!(max_abs_diff(&a, &b) < 1e-6, "{id:?}");
        }
    }

    /// Two SGD steps on the LoRA factors lower the next-token loss.
    #[test]
    fn sgd_steps_reduce_loss() {
        let (client, device) = cpu_setup();
        let mut values = initial_factors(&tiny_model(&device, 0x3a7e_0130), 0x3a7e_0131);
        let mut losses = Vec::new();
        for step in 0..3 {
            let model = adapted(tiny_model(&device, 0x3a7e_0130), &values, &device);
            let value = loss(&client, &device, &model, step == 1);
            losses.push(value.tensor().to_vec::<f32>()[0]);
            if step == 2 {
                break;
            }
            let grads = backward(&value, &client).unwrap();
            let trainable = model.trainable_parameters();
            assert_eq!(trainable.len(), values.len());
            for ((id, _), value) in trainable.iter().zip(values.iter_mut()) {
                let g = grad_values(&grads, *id);
                assert!(g.iter().all(|v| v.is_finite()));
                for (p, gv) in value.iter_mut().zip(&g) {
                    *p -= (LR * f64::from(*gv)) as f32;
                }
            }
        }
        assert!(
            losses[1] < losses[0] && losses[2] < losses[1],
            "loss must fall: {losses:?}"
        );
    }
}

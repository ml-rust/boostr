//! `MiniCpm4Model` — VoxCPM2's MiniCPM4 decoder-only transformer (`base_lm`),
//! full-sequence causal forward, inference only.
//!
//! ```text
//! inputs_embeds [B, S, hidden]
//!   -> 28x pre-norm decoder layer (causal GQA 16/2, head_dim 128, SwiGLU)
//!   -> final RmsNorm                                   [B, S, hidden]
//! ```
//!
//! Two things this model deliberately does NOT do:
//!
//! - **No `lm_head`.** The checkpoint has none. [`MiniCpm4Model::forward`]
//!   returns hidden states, never logits.
//! - **No KV cache on this path.** [`forward`](MiniCpm4Model::forward)
//!   recomputes every position on every call. The incremental (KV-cached)
//!   decode path — `new_kv_cache` / `prefill` / `decode_step` — lives in the
//!   sibling [`decode`](crate::model::audio::voxcpm::minicpm4::decode) module
//!   and leaves this one untouched.
//!
//! - **RoPE is OPTIONAL.** `residual_lm` runs NoPE (`no_rope`), so `rope` is
//!   `None` there and every attention block skips the rotation on both the
//!   full-sequence and the KV-cached path. Nothing takes its place.
//!
//! [`forward`](MiniCpm4Model::forward) takes pre-computed `inputs_embeds`
//! rather than token ids, matching the real pipeline (which feeds a combined
//! text+audio embedding). The `embed_tokens` table is exposed separately as
//! [`MiniCpm4Model::embed`] and is OPTIONAL: VoxCPM2's `residual_lm` is this
//! same architecture with `vocab_size` 0 and no table at all.
//!
//! Built from plain [`Var<R>`]-wrapped weights (`requires_grad = false`)
//! rather than autograd-tracked training params — same inference-only posture
//! as the `local_encoder` and AudioVAE siblings.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantEmbedding, Module, RmsNorm, RoPE, child_params, extend_named};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub struct MiniCpm4Model<R: Runtime> {
    /// `[vocab_size, hidden_size]` lookup table. `None` when the config
    /// carries `vocab_size == 0` (`residual_lm`), which has no such tensor in
    /// the checkpoint. May be block-quantized (packed on device, dequantized
    /// only for the rows a forward pass gathers) or standard, depending on
    /// what the checkpoint stores — see [`MaybeQuantEmbedding`].
    pub(crate) embed_tokens: Option<MaybeQuantEmbedding<R>>,
    pub(crate) layers: Vec<MiniCpm4Layer<R>>,
    pub(crate) norm: RmsNorm<R>,
    /// Precomputed cos/sin tables. `None` when the config carries
    /// `no_rope` (`residual_lm`): that stack runs NoPE, so the loader builds no
    /// table rather than a `max_position_embeddings`-row cache nothing reads.
    pub(crate) rope: Option<RoPE<R>>,
    pub(crate) hidden_size: usize,
}

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Model width.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Whether this instantiation owns an `embed_tokens` table.
    pub fn has_embedding(&self) -> bool {
        self.embed_tokens.is_some()
    }

    /// Whether this instantiation rotates Q/K.
    ///
    /// `false` for a NoPE (`no_rope`) stack, which owns no RoPE table. Every
    /// attention block in such a stack carries the matching `no_rope` flag, so
    /// no path here can reach a rotation with nothing to rotate by.
    pub fn uses_rope(&self) -> bool {
        self.rope.is_some()
    }

    /// `embed_tokens` lookup: integer `ids` `[...]` -> `[..., hidden_size]`.
    ///
    /// The result is NOT scaled: `scale_emb` (12.0) applies only under muP,
    /// which is off on this checkpoint, so the reference leaves the lookup
    /// untouched and so does this.
    ///
    /// Errors when the config had `vocab_size == 0` — a `residual_lm`-shaped
    /// instantiation is fed pre-computed embeddings and has no table to read.
    pub fn embed<C>(&self, client: &C, ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        let table = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "ids",
                reason: "this MiniCPM4 instantiation has no embed_tokens table \
                         (config vocab_size == 0); pass inputs_embeds to forward instead"
                    .to_string(),
            })?;
        table.forward(client, ids)
    }

    /// Full-sequence causal forward over pre-computed embeddings.
    ///
    /// `inputs_embeds: [batch, seq, hidden_size]` -> `[batch, seq,
    /// hidden_size]` after the final `norm`. `seq` may not exceed the RoPE
    /// cache length the loader built (`max_position_embeddings`); a NoPE
    /// (`no_rope`) stack has no such cache and no such bound.
    pub fn forward<C>(&self, client: &C, inputs_embeds: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
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
        let shape = inputs_embeds.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected 3D [batch, seq, hidden_size], got {}D",
                    shape.len()
                ),
            });
        }
        if shape[2] != self.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected hidden_size {}, got {}",
                    self.hidden_size, shape[2]
                ),
            });
        }

        let mut h = inputs_embeds.clone();
        for layer in &self.layers {
            h = layer.forward(client, &h, self.rope.as_ref())?;
        }
        self.norm.forward(client, &h)
    }
}

/// Names ARE the field names (`embed_tokens`, `layers.{i}.*`, `norm`) —
/// this matches the `{prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so
/// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model) need
/// only prefix by `base_lm`/`residual_lm` to reach the full checkpoint key.
/// `embed_tokens` is absent entirely on a `residual_lm` instantiation
/// (`vocab_size == 0`) — correctly contributing nothing rather than a
/// zero-filled placeholder. `rope` carries no `Var<R>` (a precomputed,
/// non-learned cos/sin cache, like every other `RoPE` table in this crate)
/// and is correctly absent below.
impl<R: Runtime> Module<R> for MiniCpm4Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = self
            .embed_tokens
            .as_ref()
            .map(|embed_tokens| child_params(embed_tokens))
            .unwrap_or_default();
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(embed_tokens) = &self.embed_tokens {
            extend_named(&mut params, "embed_tokens", embed_tokens.named_parameters());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(&mut params, "norm", self.norm.named_parameters());
        params
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
    use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
    use crate::nn::{MaybeQuantLinear, Weight};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    pub(crate) const HIDDEN: usize = 8;
    const NUM_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 4;
    const FFN: usize = 16;
    const NUM_LAYERS: usize = 2;

    /// Deterministic, non-degenerate weights: zeros would make every
    /// causality/shape assertion below pass vacuously.
    pub(crate) fn filled(shape: &[usize], salt: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + salt * 11) % 13) as f32 - 6.0) / 20.0)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("weights")
    }

    fn linear(
        out: usize,
        in_dim: usize,
        salt: usize,
        device: &CpuDevice,
    ) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::from_weight(Weight::Standard(filled(&[out, in_dim], salt, device)), None)
    }

    /// `base_lm`-shaped tiny model: rotary, exactly as before `no_rope`
    /// existed.
    pub(crate) fn tiny_model(device: &CpuDevice) -> MiniCpm4Model<CpuRuntime> {
        tiny_model_with(device, false)
    }

    /// `residual_lm`-shaped tiny model: NoPE, no RoPE table at all.
    pub(crate) fn tiny_nope_model(device: &CpuDevice) -> MiniCpm4Model<CpuRuntime> {
        tiny_model_with(device, true)
    }

    /// Same weights either way, so any output difference between the two is
    /// the rotation and nothing else.
    fn tiny_model_with(device: &CpuDevice, no_rope: bool) -> MiniCpm4Model<CpuRuntime> {
        let q_dim = NUM_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let layers = (0..NUM_LAYERS)
            .map(|i| MiniCpm4Layer {
                input_layernorm: RmsNorm::new(
                    Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
                    1e-5,
                    false,
                ),
                self_attn: MiniCpm4Attention {
                    q_proj: linear(q_dim, HIDDEN, i * 8 + 1, device),
                    k_proj: linear(kv_dim, HIDDEN, i * 8 + 2, device),
                    v_proj: linear(kv_dim, HIDDEN, i * 8 + 3, device),
                    o_proj: linear(HIDDEN, q_dim, i * 8 + 4, device),
                    num_heads: NUM_HEADS,
                    num_kv_heads: NUM_KV_HEADS,
                    head_dim: HEAD_DIM,
                    no_rope,
                },
                post_attention_layernorm: RmsNorm::new(
                    Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
                    1e-5,
                    false,
                ),
                mlp: MiniCpm4Mlp {
                    gate_proj: linear(FFN, HIDDEN, i * 8 + 5, device),
                    up_proj: linear(FFN, HIDDEN, i * 8 + 6, device),
                    down_proj: linear(HIDDEN, FFN, i * 8 + 7, device),
                },
            })
            .collect();
        MiniCpm4Model {
            embed_tokens: None,
            layers,
            norm: RmsNorm::new(
                Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
                1e-5,
                false,
            ),
            rope: (!no_rope).then(|| {
                RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, device)
                    .expect("rope")
            }),
            hidden_size: HIDDEN,
        }
    }

    fn out_values(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let x = Var::new(filled(&[1, 4, HIDDEN], 99, &device), false);
        let out = model.forward(&client, &x).expect("forward");
        assert_eq!(out.shape(), &[1, 4, HIDDEN]);
        assert!(out_values(&out).iter().all(|v| v.is_finite()));
    }

    /// The load-bearing property of this port: attention is CAUSAL. Perturbing
    /// the LAST position must leave every earlier output bit-identical.
    #[test]
    fn attention_is_causal() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);

        let seq = 4;
        let mut data: Vec<f32> = (0..seq * HIDDEN)
            .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
            .collect();
        let base = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
            false,
        );
        let base_out = out_values(&model.forward(&client, &base).expect("forward"));

        // Perturb only the final position.
        for value in data.iter_mut().skip((seq - 1) * HIDDEN) {
            *value += 3.0;
        }
        let perturbed = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
            false,
        );
        let perturbed_out = out_values(&model.forward(&client, &perturbed).expect("forward"));

        let prefix = (seq - 1) * HIDDEN;
        assert_eq!(
            base_out[..prefix],
            perturbed_out[..prefix],
            "earlier positions changed: attention is not causal"
        );
        assert!(
            base_out[prefix..]
                .iter()
                .zip(&perturbed_out[prefix..])
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "final position did not react to its own perturbation"
        );
    }

    /// The flag has to be load-bearing: a NoPE stack must not reproduce the
    /// rotary stack's numbers on the same weights and the same input.
    #[test]
    fn nope_output_differs_from_rotary_output() {
        let (client, device) = cpu_setup();
        let rotary = tiny_model(&device);
        let nope = tiny_nope_model(&device);
        assert!(rotary.uses_rope());
        assert!(!nope.uses_rope());

        let x = Var::new(filled(&[1, 4, HIDDEN], 99, &device), false);
        let rotary_out = out_values(&rotary.forward(&client, &x).expect("forward"));
        let nope_out = out_values(&nope.forward(&client, &x).expect("forward"));

        assert!(
            rotary_out
                .iter()
                .zip(&nope_out)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "no_rope changed nothing: the flag is not reaching the attention blocks"
        );
        assert!(nope_out.iter().all(|v| v.is_finite()));
    }

    /// NoPE drops the rotation and substitutes NOTHING, but the causal mask
    /// still applies: earlier positions may not see a later one.
    #[test]
    fn nope_attention_is_still_causal() {
        let (client, device) = cpu_setup();
        let model = tiny_nope_model(&device);

        let seq = 4;
        let mut data: Vec<f32> = (0..seq * HIDDEN)
            .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
            .collect();
        let base = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
            false,
        );
        let base_out = out_values(&model.forward(&client, &base).expect("forward"));

        for value in data.iter_mut().skip((seq - 1) * HIDDEN) {
            *value += 3.0;
        }
        let perturbed = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
            false,
        );
        let perturbed_out = out_values(&model.forward(&client, &perturbed).expect("forward"));

        let prefix = (seq - 1) * HIDDEN;
        assert_eq!(
            base_out[..prefix],
            perturbed_out[..prefix],
            "earlier positions changed: NoPE attention is not causal"
        );
        assert!(
            base_out[prefix..]
                .iter()
                .zip(&perturbed_out[prefix..])
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "final position did not react to its own perturbation"
        );
    }

    #[test]
    fn rejects_wrong_hidden_size() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let x = Var::new(filled(&[1, 2, HIDDEN + 1], 5, &device), false);
        let err = model.forward(&client, &x).unwrap_err();
        assert!(err.to_string().contains("hidden_size"), "got {err}");
    }

    #[test]
    fn rejects_non_3d_input() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let x = Var::new(filled(&[2, HIDDEN], 5, &device), false);
        assert!(model.forward(&client, &x).is_err());
    }

    #[test]
    fn embed_errors_without_table() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        assert!(!model.has_embedding());
        let ids = Tensor::<CpuRuntime>::zeros(&[1, 2], DType::I64, &device).expect("ids");
        let err = model.embed(&client, &ids).unwrap_err();
        assert!(err.to_string().contains("vocab_size"), "got {err}");
    }

    /// [`MiniCpm4Model::parameters`]/[`named_parameters`] (via `Module`) on
    /// the same tiny fixture the forward-pass tests above build.
    /// `tiny_model` has NO `embed_tokens` (like the real `residual_lm` half
    /// of this model, though here for a different reason — the fixture
    /// simply never sets one), so this also pins that the absent table
    /// contributes nothing rather than a placeholder entry.
    #[test]
    fn module_enumeration_is_non_empty_with_unique_ids_and_names() {
        let (_client, device) = cpu_setup();
        let model = tiny_model(&device);

        let params = model.parameters();
        assert!(!params.is_empty());
        let ids: std::collections::HashSet<_> = params.iter().map(|v| v.id()).collect();
        assert_eq!(ids.len(), params.len(), "duplicate TensorId");

        let named = model.named_parameters();
        assert_eq!(named.len(), params.len());
        let names: std::collections::HashSet<_> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), named.len(), "duplicate parameter name");

        assert!(!named.iter().any(|(n, _)| n.starts_with("embed_tokens")));
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "layers.0.self_attn.q_proj.weight")
        );
        assert!(named.iter().any(|(n, _)| n == "norm.weight"));
    }
}

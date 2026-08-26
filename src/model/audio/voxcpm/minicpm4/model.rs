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
use crate::nn::{Embedding, RmsNorm, RoPE};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub struct MiniCpm4Model<R: Runtime> {
    /// `[vocab_size, hidden_size]` lookup table. `None` when the config
    /// carries `vocab_size == 0` (`residual_lm`), which has no such tensor in
    /// the checkpoint.
    pub(crate) embed_tokens: Option<Embedding<R>>,
    pub(crate) layers: Vec<MiniCpm4Layer<R>>,
    pub(crate) norm: RmsNorm<R>,
    pub(crate) rope: RoPE<R>,
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
        C: ModelClient<R>,
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
    /// cache length the loader built (`max_position_embeddings`).
    pub fn forward<C>(&self, client: &C, inputs_embeds: &Var<R>) -> Result<Var<R>>
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
            h = layer.forward(client, &h, &self.rope)?;
        }
        self.norm.forward(client, &h)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
    use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
    use crate::nn::Linear;
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

    fn linear(out: usize, in_dim: usize, salt: usize, device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(filled(&[out, in_dim], salt, device), None, false)
    }

    pub(crate) fn tiny_model(device: &CpuDevice) -> MiniCpm4Model<CpuRuntime> {
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
            rope: RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, device)
                .expect("rope"),
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
}

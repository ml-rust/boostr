//! [`Qwen35Model`] forward passes and accessors.
//!
//! The layer walk keeps two counters, `attn_idx` into the [`LayeredKvCache`]
//! and `gdn_idx` into the [`LayeredGdnState`], exactly as
//! `HybridModel::forward_hybrid` does: each state container is indexed by
//! position among layers of its kind, not by absolute layer index.

use super::build::{Qwen35Block, Qwen35Model};
use crate::error::{Error, Result};
use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::config::{GdnConfig, HybridConfig, Qwen35AttentionConfig, UniversalConfig};
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE, RotatedMlp};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Inference forward: logits `[batch, seq, vocab_size]`.
    ///
    /// `rope_pos` is the IMROPE position of `input_ids[.., 0]`; token `i`
    /// sits at `rope_pos + i` on the `t`, `h` and `w` streams and `0` on
    /// `e`. The KV slot comes from `kv_cache.seq_len()`, so after an image
    /// the two counters differ and the caller passes its own rope counter.
    /// Text-only callers pass `kv_cache.seq_len()`.
    ///
    /// The token path is [`embed_tokens`](Self::embed_tokens) followed by
    /// [`forward_qwen35_embeds`](Self::forward_qwen35_embeds).
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `input_ids` is not `[batch, seq]`, when
    /// `rope_pos + seq` exceeds the rope table, or when `kv_cache` /
    /// `gdn_state` hold fewer layers than the model has of that kind. Block
    /// errors propagate.
    pub fn forward_qwen35<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        rope_pos: usize,
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
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::ModelError {
                reason: format!("qwen35: input_ids must be [batch, seq], got {shape:?}"),
            });
        }
        let positions = self.text_positions(shape[1], rope_pos, input_ids.device())?;
        let embeds = self.embed_tokens(client, input_ids)?;
        self.forward_qwen35_embeds(client, &embeds, &positions, kv_cache, gdn_state)
    }

    /// Contextualized hidden states for embedding extraction.
    ///
    /// Runs embed + every block + `output_norm` (no `lm_head`) over fresh
    /// throwaway KV cache and GDN state. Returns `[batch, seq, hidden]`.
    pub fn forward_hidden<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
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
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::ModelError {
                reason: format!("qwen35: input_ids must be [batch, seq], got {shape:?}"),
            });
        }
        let (batch, seq_len) = (shape[0], shape[1]);
        let device = input_ids.device();
        let dtype = self.norm.weight().tensor().dtype();

        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: self.attention_config.num_kv_heads,
            initial_capacity: seq_len,
            max_seq_len: self.config.max_seq_len,
            head_dim: self.attention_config.head_dim,
            dtype,
        };
        let mut kv_cache =
            LayeredKvCache::<R>::new(self.num_attention_layers(), &kv_config, device)?;
        let mut gdn_state = LayeredGdnState::<R>::zeros(
            self.num_gdn_layers(),
            &self.gdn_config,
            batch,
            dtype,
            device,
        )?;
        let positions = self.text_positions(seq_len, 0, device)?;
        let hidden = self.embed_tokens.forward(client, input_ids)?;
        self.forward_layers(client, hidden, &positions, &mut kv_cache, &mut gdn_state)
    }

    /// Walk every block over `hidden` `[batch, seq, hidden]`, then apply
    /// `output_norm`. `positions` is the `[4, seq]` i32 IMROPE tensor every
    /// attention layer reads.
    pub(super) fn forward_layers<C>(
        &self,
        client: &C,
        hidden: Var<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<Var<R>>
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
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let mut hidden = hidden;
        let mut attn_idx = 0usize;
        let mut gdn_idx = 0usize;

        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                Qwen35Block::Attention(layer) => {
                    let cache = kv_cache
                        .layer_mut(attn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "qwen35: KV cache missing for layer {i} (attn_idx={attn_idx})"
                            ),
                        })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = layer
                        .mixer
                        .forward(client, &normed, &self.rope, positions, cache)?;
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    attn_idx += 1;
                }
                Qwen35Block::Gdn(layer) => {
                    let state = gdn_state
                        .layer_mut(gdn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "qwen35: GDN state missing for layer {i} (gdn_idx={gdn_idx})"
                            ),
                        })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = layer.mixer.forward(client, &normed, state)?;
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    gdn_idx += 1;
                }
            }
        }

        self.norm.forward(client, &hidden)
    }

    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    pub fn gdn_config(&self) -> &GdnConfig {
        &self.gdn_config
    }

    pub fn attention_config(&self) -> &Qwen35AttentionConfig {
        &self.attention_config
    }

    /// Per-layer kind assignment.
    pub fn layers(&self) -> &HybridConfig {
        &self.layers
    }

    /// GDN layer count (for GDN state allocation).
    pub fn num_gdn_layers(&self) -> usize {
        self.layers.ssm_layers.len()
    }

    /// Attention layer count (for KV cache allocation).
    pub fn num_attention_layers(&self) -> usize {
        self.layers.attention_layers.len()
    }

    /// The IMROPE cos/sin table, `[max_seq_len, rope_dim / 2]`.
    pub fn rope(&self) -> &RoPE<R> {
        &self.rope
    }
}

/// `h = x + mixed; h + mlp(post_norm(h))`.
pub(super) fn ffn_residual<R, C>(
    client: &C,
    x: &Var<R>,
    mixed: &Var<R>,
    post_norm: &RmsNorm<R>,
    mlp: &RotatedMlp<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + DequantOps<R>
        + MatmulOps<R>,
{
    let h = var_add(x, mixed, client).map_err(Error::Numr)?;
    let normed = post_norm.forward(client, &h)?;
    let ffn = mlp.forward(client, &normed)?;
    var_add(&h, &ffn, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{
        H_KV, HD, HIDDEN, MAX_POS, SEQ, VOCAB, rotated_model, tiny_model,
    };
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn caches(
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
    ) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: H_KV,
            initial_capacity: MAX_POS,
            max_seq_len: MAX_POS,
            head_dim: HD,
            dtype: DType::F32,
        };
        let kv =
            LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
                .unwrap();
        let gdn = LayeredGdnState::<CpuRuntime>::zeros(
            model.num_gdn_layers(),
            model.gdn_config(),
            1,
            DType::F32,
            device,
        )
        .unwrap();
        (kv, gdn)
    }

    fn tokens(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (0..SEQ).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[1, SEQ], device).unwrap()
    }

    fn run(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        ids: &Tensor<CpuRuntime>,
        kv: &mut LayeredKvCache<CpuRuntime>,
        gdn: &mut LayeredGdnState<CpuRuntime>,
        position: usize,
    ) -> Tensor<CpuRuntime> {
        model
            .forward_qwen35(client, ids, kv, gdn, position)
            .unwrap()
    }

    fn max_abs_diff(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
        let a = a.to_vec::<f32>();
        let b = b.to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn logits_shape() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0010);
        let (mut kv, mut gdn) = caches(&device, &model);
        let logits = run(&client, &model, &tokens(&device), &mut kv, &mut gdn, 0);
        assert_eq!(logits.shape(), &[1, SEQ, VOCAB]);
        assert!(logits.to_vec::<f32>().iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq_len(), SEQ);
    }

    /// Prefill of 7 tokens equals prefill of 4 then 3 decode steps through
    /// the same KV cache and GDN state.
    #[test]
    fn prefill_matches_prefill_then_decode() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0011);
        let ids = tokens(&device);

        let (mut kv, mut gdn) = caches(&device, &model);
        let full = run(&client, &model, &ids, &mut kv, &mut gdn, 0);

        let (mut kv, mut gdn) = caches(&device, &model);
        let head = ids.narrow(1, 0, 4).unwrap().contiguous().unwrap();
        let mut parts = vec![run(&client, &model, &head, &mut kv, &mut gdn, 0)];
        for t in 4..SEQ {
            let step = ids.narrow(1, t, 1).unwrap().contiguous().unwrap();
            parts.push(run(&client, &model, &step, &mut kv, &mut gdn, t));
        }
        let refs: Vec<&Tensor<CpuRuntime>> = parts.iter().collect();
        let stepped = client.cat(&refs, 1).unwrap();

        let diff = max_abs_diff(&full, &stepped);
        assert!(diff < 1e-4, "logits diff {diff}");
    }

    #[test]
    fn rope_pos_beyond_table_is_an_error() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0012);
        let (mut kv, mut gdn) = caches(&device, &model);
        let ids = tokens(&device);
        assert!(
            model
                .forward_qwen35(&client, &ids, &mut kv, &mut gdn, MAX_POS - SEQ + 1)
                .is_err()
        );
    }

    /// The rope position is independent of the KV slot: decoding at a rope
    /// position ahead of the cache length runs and differs from the aligned
    /// decode.
    #[test]
    fn rope_pos_decoupled_from_kv_slot() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0015);
        let ids = tokens(&device);
        let head = ids.narrow(1, 0, 4).unwrap().contiguous().unwrap();
        let step = ids.narrow(1, 4, 1).unwrap().contiguous().unwrap();

        let (mut kv_a, mut gdn_a) = caches(&device, &model);
        run(&client, &model, &head, &mut kv_a, &mut gdn_a, 0);
        let aligned = run(&client, &model, &step, &mut kv_a, &mut gdn_a, 4);

        let (mut kv_b, mut gdn_b) = caches(&device, &model);
        run(&client, &model, &head, &mut kv_b, &mut gdn_b, 0);
        let shifted = run(&client, &model, &step, &mut kv_b, &mut gdn_b, 7);

        assert_eq!(kv_a.seq_len(), 5);
        assert_eq!(kv_b.seq_len(), 5);
        assert!(max_abs_diff(&aligned, &shifted) > 1e-6);
    }

    /// Same weights wrapped as `Rotated` linears run, and the output differs
    /// from the `Plain` wrapping — the rotation is applied, not skipped.
    #[test]
    fn rotated_linears_change_output() {
        let (client, device) = cpu_setup();
        let plain = tiny_model(&device, 0x3535_0013);
        let rotated = rotated_model(&device, 0x3535_0013);
        let ids = tokens(&device);
        let (mut kv_a, mut gdn_a) = caches(&device, &plain);
        let (mut kv_b, mut gdn_b) = caches(&device, &rotated);
        let a = run(&client, &plain, &ids, &mut kv_a, &mut gdn_a, 0);
        let b = run(&client, &rotated, &ids, &mut kv_b, &mut gdn_b, 0);
        assert_eq!(b.shape(), &[1, SEQ, VOCAB]);
        assert!(b.to_vec::<f32>().iter().all(|v| v.is_finite()));
        assert!(max_abs_diff(&a, &b) > 1e-4);
    }

    #[test]
    fn forward_hidden_matches_prefill_before_lm_head() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0014);
        let ids = tokens(&device);
        let hidden = model.forward_hidden(&client, &ids).unwrap();
        assert_eq!(hidden.shape(), &[1, SEQ, HIDDEN]);

        let (mut kv, mut gdn) = caches(&device, &model);
        let logits = run(&client, &model, &ids, &mut kv, &mut gdn, 0);
        let from_hidden = model.lm_head.forward(&client, &hidden).unwrap();
        assert!(max_abs_diff(&logits, from_hidden.tensor()) < 1e-5);
    }
}

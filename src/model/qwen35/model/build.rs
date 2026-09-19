//! The [`Qwen35Model`] type, its layer structs, and [`Qwen35Model::new`].
//!
//! Layer structure, matching the `qwen35.cpp` graph:
//!
//! ```text
//! h   = x + mixer(attn_norm(x))            // mixer: GDN or gated attention
//! out = h + ffn(post_attention_norm(h))    // ffn: SwiGLU
//! ```
//!
//! then `output_norm` and `lm_head` once after the last layer.
//!
//! The GGUF loader is a separate unit: `new` takes built modules so the
//! Hadamard attach and the quantized-vs-dense choice stay with the loader.

use crate::error::{Error, Result};
use crate::model::config::{GdnConfig, HybridConfig, Qwen35AttentionConfig, UniversalConfig};
use crate::model::hybrid::{GdnBlock, Qwen35AttentionBlock};
use crate::nn::{MaybeRotatedEmbedding, MaybeRotatedLinear, RmsNorm, RoPE, RotatedMlp};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// A GDN (linear-attention) layer: pre-norm, mixer, post-norm, FFN.
pub struct Qwen35GdnLayer<R: Runtime> {
    pub attn_norm: RmsNorm<R>,
    pub mixer: GdnBlock<R>,
    pub post_attention_norm: RmsNorm<R>,
    pub mlp: RotatedMlp<R>,
}

/// A full-attention layer: pre-norm, mixer, post-norm, FFN.
pub struct Qwen35AttentionLayer<R: Runtime> {
    pub attn_norm: RmsNorm<R>,
    pub mixer: Qwen35AttentionBlock<R>,
    pub post_attention_norm: RmsNorm<R>,
    pub mlp: RotatedMlp<R>,
}

/// One layer of the model, by mixer kind.
pub enum Qwen35Block<R: Runtime> {
    Gdn(Box<Qwen35GdnLayer<R>>),
    Attention(Box<Qwen35AttentionLayer<R>>),
}

impl<R: Runtime> Qwen35Block<R> {
    /// `true` for an attention layer.
    pub fn is_attention(&self) -> bool {
        matches!(self, Self::Attention(_))
    }
}

/// `qwen35` model: embed → `Qwen35Block`s → `output_norm` → `lm_head`.
///
/// `config` is the [`UniversalConfig`] that describes the model; `gdn`,
/// `qwen35_attention` and `hybrid_layers` must all be set. The three are
/// cloned out at construction so accessors return them without an `Option`.
pub struct Qwen35Model<R: Runtime> {
    pub(super) config: UniversalConfig,
    pub(super) gdn_config: GdnConfig,
    pub(super) attention_config: Qwen35AttentionConfig,
    pub(super) layers: HybridConfig,
    pub(super) embed_tokens: MaybeRotatedEmbedding<R>,
    pub(super) blocks: Vec<Qwen35Block<R>>,
    pub(super) norm: RmsNorm<R>,
    pub(super) lm_head: MaybeRotatedLinear<R>,
    pub(super) rope: RoPE<R>,
}

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// Assemble from built modules.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `config` fails validation or lacks `gdn`,
    /// `qwen35_attention` or `hybrid_layers`; when `blocks` does not match
    /// `hybrid_layers` in count or per-index kind; when `embed_tokens` is
    /// not `[vocab_size, hidden_size]`, `lm_head` is not
    /// `[vocab_size, hidden_size]`, or `rope` is not `[_, rope_dim / 2]`.
    pub fn new(
        config: UniversalConfig,
        embed_tokens: MaybeRotatedEmbedding<R>,
        blocks: Vec<Qwen35Block<R>>,
        norm: RmsNorm<R>,
        lm_head: MaybeRotatedLinear<R>,
        rope: RoPE<R>,
    ) -> Result<Self> {
        config.validate()?;
        let gdn_config = config.gdn.clone().ok_or_else(|| Error::ModelError {
            reason: "qwen35 requires a gdn config".into(),
        })?;
        let attention_config =
            config
                .qwen35_attention
                .clone()
                .ok_or_else(|| Error::ModelError {
                    reason: "qwen35 requires a qwen35_attention config".into(),
                })?;
        let layers = config
            .hybrid_layers
            .clone()
            .ok_or_else(|| Error::ModelError {
                reason: "qwen35 requires hybrid_layers (attention vs gdn per layer)".into(),
            })?;

        if blocks.len() != config.num_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: {} blocks for num_layers = {}",
                    blocks.len(),
                    config.num_layers
                ),
            });
        }
        for (i, block) in blocks.iter().enumerate() {
            let want_attention = layers.is_attention_layer(i);
            if block.is_attention() != want_attention {
                let (want, got) = if want_attention {
                    ("attention", "gdn")
                } else {
                    ("gdn", "attention")
                };
                return Err(Error::ModelError {
                    reason: format!("qwen35: layer {i} is {got}, hybrid_layers says {want}"),
                });
            }
        }

        let (vocab, hidden) = (config.vocab_size, config.hidden_size);
        let embed_shape = [embed_tokens.num_embeddings(), embed_tokens.embedding_dim()];
        if embed_shape != [vocab, hidden] {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: embed_tokens is {embed_shape:?}, expected [{vocab}, {hidden}]"
                ),
            });
        }
        let head_shape = lm_head.base().shape();
        if head_shape != [vocab, hidden] {
            return Err(Error::ModelError {
                reason: format!("qwen35: lm_head is {head_shape:?}, expected [{vocab}, {hidden}]"),
            });
        }
        let half_rot = attention_config.rope_dim / 2;
        if rope.cos_cache().shape().get(1) != Some(&half_rot) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: rope table must be [max_pos, {half_rot}], got {:?}",
                    rope.cos_cache().shape()
                ),
            });
        }

        Ok(Self {
            config,
            gdn_config,
            attention_config,
            layers,
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::hybrid::{GdnWeights, Qwen35AttentionWeights};
    use crate::nn::{
        Embedding, HadamardRotation, Linear, MaybeQuantEmbedding, MaybeQuantLinear, RotatedLinear,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::collections::HashMap;

    pub(crate) const HIDDEN: usize = 8;
    pub(crate) const VOCAB: usize = 16;
    pub(crate) const MAX_POS: usize = 32;
    const INTER: usize = 16;
    const H_K: usize = 2;
    const H_V: usize = 4;
    const S: usize = 4;
    const KERNEL: usize = 4;
    const H: usize = 2;
    pub(crate) const H_KV: usize = 1;
    pub(crate) const HD: usize = 8;
    const ROPE_DIM: usize = 4;
    pub(crate) const SEQ: usize = 7;
    const ROT_BLOCK: usize = 4;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }

        fn tensor(
            &mut self,
            device: &CpuDevice,
            shape: &[usize],
            scale: f32,
        ) -> Tensor<CpuRuntime> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| self.uniform(-scale, scale)).collect();
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
        }
    }

    pub(crate) fn tiny_config() -> UniversalConfig {
        UniversalConfig {
            model_type: "qwen35".into(),
            vocab_size: VOCAB,
            hidden_size: HIDDEN,
            num_layers: 2,
            max_seq_len: MAX_POS,
            intermediate_size: Some(INTER),
            rms_norm_eps: 1e-6,
            attention: None,
            ssm: None,
            moe: None,
            hybrid_layers: Some(HybridConfig::from_full_attention_interval(2, 2)),
            tie_word_embeddings: false,
            grow_vocab: false,
            vision: None,
            audio: None,
            gdn: Some(GdnConfig {
                hidden_size: HIDDEN,
                conv_kernel: KERNEL,
                state_size: S,
                key_heads: H_K,
                value_heads: H_V,
                inner_size: H_V * S,
                rms_eps: 1e-6,
                chunk_size: 64,
                v_grouped: false,
            }),
            qwen35_attention: Some(Qwen35AttentionConfig {
                hidden_size: HIDDEN,
                num_heads: H,
                num_kv_heads: H_KV,
                head_dim: HD,
                rope_dim: ROPE_DIM,
                rope_sections: [1, 1, 0, 0],
                rope_theta: 10_000.0,
                rms_eps: 1e-6,
            }),
            hadamard: None,
        }
    }

    /// Weights and modules that `Qwen35Model::new` takes.
    pub(crate) type Parts = (
        UniversalConfig,
        MaybeRotatedEmbedding<CpuRuntime>,
        Vec<Qwen35Block<CpuRuntime>>,
        RmsNorm<CpuRuntime>,
        MaybeRotatedLinear<CpuRuntime>,
        RoPE<CpuRuntime>,
    );

    struct Builder<'a> {
        device: &'a CpuDevice,
        rng: Lcg,
        rotated: bool,
        /// One rotation per input width, handed out as clones, as
        /// `Attach::rotation` does for a real file: layers that share an
        /// input share the signs storage, which `forward_batch` checks.
        rotations: HashMap<usize, HadamardRotation<CpuRuntime>>,
    }

    impl Builder<'_> {
        fn rotation(&mut self, inp: usize) -> HadamardRotation<CpuRuntime> {
            if let Some(r) = self.rotations.get(&inp) {
                return r.clone();
            }
            let signs: Vec<i8> = (0..inp).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
            let rotation = HadamardRotation::<CpuRuntime>::new(
                ROT_BLOCK,
                Some(&signs),
                DType::F32,
                self.device,
            )
            .unwrap();
            self.rotations.insert(inp, rotation.clone());
            rotation
        }

        fn linear(&mut self, out: usize, inp: usize) -> MaybeRotatedLinear<CpuRuntime> {
            let w = self
                .rng
                .tensor(self.device, &[out, inp], 0.5 / (inp as f32).sqrt());
            let base = MaybeQuantLinear::Standard(Linear::new(w, None, false));
            if !self.rotated {
                return MaybeRotatedLinear::Plain(base);
            }
            let rotation = self.rotation(inp);
            MaybeRotatedLinear::Rotated(Box::new(RotatedLinear::new(base, rotation).unwrap()))
        }

        fn plain(&mut self, out: usize, inp: usize) -> MaybeQuantLinear<CpuRuntime> {
            let w = self
                .rng
                .tensor(self.device, &[out, inp], 0.5 / (inp as f32).sqrt());
            MaybeQuantLinear::Standard(Linear::new(w, None, false))
        }

        fn norm(&mut self) -> RmsNorm<CpuRuntime> {
            let w: Vec<f32> = (0..HIDDEN).map(|_| self.rng.uniform(0.5, 1.5)).collect();
            let w = Tensor::<CpuRuntime>::from_slice(&w, &[HIDDEN], self.device).unwrap();
            RmsNorm::new(w, 1e-6, false)
        }

        fn mlp(&mut self) -> RotatedMlp<CpuRuntime> {
            let gate = self.linear(INTER, HIDDEN);
            let up = self.linear(INTER, HIDDEN);
            let down = self.linear(HIDDEN, INTER);
            RotatedMlp::new(gate, up, down).unwrap()
        }

        fn gdn(&mut self, cfg: &GdnConfig) -> GdnBlock<CpuRuntime> {
            let a: Vec<f32> = (0..H_V)
                .map(|_| -self.rng.uniform(0.5, 1.5).exp())
                .collect();
            let weights = GdnWeights {
                attn_qkv: self.linear(cfg.qkv_dim(), HIDDEN),
                attn_gate: self.linear(cfg.value_dim(), HIDDEN),
                ssm_alpha: self.plain(H_V, HIDDEN),
                ssm_beta: self.plain(H_V, HIDDEN),
                ssm_out: self.linear(HIDDEN, cfg.value_dim()),
                ssm_conv1d: self.rng.tensor(self.device, &[cfg.qkv_dim(), KERNEL], 0.5),
                ssm_a: Tensor::<CpuRuntime>::from_slice(&a, &[H_V], self.device).unwrap(),
                ssm_dt_bias: self.rng.tensor(self.device, &[H_V], 0.5),
                ssm_norm: self.rng.tensor(self.device, &[S], 1.0),
            };
            GdnBlock::new(cfg.clone(), weights).unwrap()
        }

        fn attention(&mut self, cfg: &Qwen35AttentionConfig) -> Qwen35AttentionBlock<CpuRuntime> {
            let q_norm: Vec<f32> = (0..HD).map(|_| self.rng.uniform(0.5, 1.5)).collect();
            let weights = Qwen35AttentionWeights {
                attn_q: self.linear(cfg.q_gate_dim(), HIDDEN),
                attn_k: self.linear(cfg.kv_dim(), HIDDEN),
                attn_v: self.linear(cfg.kv_dim(), HIDDEN),
                attn_output: self.linear(HIDDEN, cfg.q_dim()),
                attn_q_norm: Tensor::<CpuRuntime>::from_slice(&q_norm, &[HD], self.device).unwrap(),
                attn_k_norm: self.rng.tensor(self.device, &[HD], 1.0),
            };
            Qwen35AttentionBlock::new(cfg.clone(), weights).unwrap()
        }
    }

    fn parts(device: &CpuDevice, seed: u64, rotated: bool) -> Parts {
        let config = tiny_config();
        let gdn_cfg = config.gdn.clone().unwrap();
        let attn_cfg = config.qwen35_attention.clone().unwrap();
        let mut b = Builder {
            device,
            rng: Lcg(seed),
            rotated,
            rotations: HashMap::new(),
        };
        let table = b.rng.tensor(device, &[VOCAB, HIDDEN], 1.0);
        let embed = MaybeRotatedEmbedding::Plain(MaybeQuantEmbedding::Standard(Embedding::new(
            table, false,
        )));
        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            let attn_norm = b.norm();
            let post_attention_norm = b.norm();
            let mlp = b.mlp();
            if config.hybrid_layers.as_ref().unwrap().is_attention_layer(i) {
                let mixer = b.attention(&attn_cfg);
                blocks.push(Qwen35Block::Attention(Box::new(Qwen35AttentionLayer {
                    attn_norm,
                    mixer,
                    post_attention_norm,
                    mlp,
                })));
            } else {
                let mixer = b.gdn(&gdn_cfg);
                blocks.push(Qwen35Block::Gdn(Box::new(Qwen35GdnLayer {
                    attn_norm,
                    mixer,
                    post_attention_norm,
                    mlp,
                })));
            }
        }
        let norm = b.norm();
        let lm_head = b.linear(VOCAB, HIDDEN);
        let rope = attn_cfg.rope_table::<CpuRuntime>(MAX_POS, device).unwrap();
        (config, embed, blocks, norm, lm_head, rope)
    }

    pub(crate) fn tiny_parts(device: &CpuDevice, seed: u64) -> Parts {
        parts(device, seed, false)
    }

    pub(crate) fn tiny_model(device: &CpuDevice, seed: u64) -> Qwen35Model<CpuRuntime> {
        let (config, embed, blocks, norm, lm_head, rope) = parts(device, seed, false);
        Qwen35Model::new(config, embed, blocks, norm, lm_head, rope).unwrap()
    }

    pub(crate) fn rotated_model(device: &CpuDevice, seed: u64) -> Qwen35Model<CpuRuntime> {
        let (config, embed, blocks, norm, lm_head, rope) = parts(device, seed, true);
        Qwen35Model::new(config, embed, blocks, norm, lm_head, rope).unwrap()
    }

    #[test]
    fn rejects_layer_kind_mismatch() {
        let (_client, device) = cpu_setup();
        let (config, embed, mut blocks, norm, lm_head, rope) = tiny_parts(&device, 0x3535_0001);
        blocks.swap(0, 1);
        assert!(Qwen35Model::new(config, embed, blocks, norm, lm_head, rope).is_err());
    }

    #[test]
    fn rejects_block_count_mismatch() {
        let (_client, device) = cpu_setup();
        let (config, embed, mut blocks, norm, lm_head, rope) = tiny_parts(&device, 0x3535_0002);
        blocks.pop();
        assert!(Qwen35Model::new(config, embed, blocks, norm, lm_head, rope).is_err());
    }

    #[test]
    fn accessors_report_layer_split() {
        let (_client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0003);
        assert_eq!(model.num_gdn_layers(), 1);
        assert_eq!(model.num_attention_layers(), 1);
        assert_eq!(model.config().hidden_size, HIDDEN);
        assert_eq!(model.gdn_config().hidden_size, HIDDEN);
        assert_eq!(model.attention_config().hidden_size, HIDDEN);
    }
}

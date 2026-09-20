//! A tiny synthetic `qwen35` hybrid model on any runtime, built from the
//! public constructors. Deterministic weights from a seed; every linear
//! carries a Hadamard rotation, layers that share an input share one.
//!
//! Head dim is 32, the smallest the CUDA graph decode kernel accepts, so
//! the same model serves eager and graph-mode tests.

use std::collections::HashMap;

use boostr::model::config::{GdnConfig, HybridConfig, Qwen35AttentionConfig, UniversalConfig};
use boostr::model::hybrid::{GdnBlock, GdnWeights, Qwen35AttentionBlock, Qwen35AttentionWeights};
use boostr::model::qwen35::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
use boostr::nn::{
    Embedding, HadamardRotation, Linear, MaybeQuantEmbedding, MaybeQuantLinear,
    MaybeRotatedEmbedding, MaybeRotatedLinear, RmsNorm, RotatedLinear, RotatedMlp,
};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub const HIDDEN: usize = 8;
pub const VOCAB: usize = 16;
pub const MAX_POS: usize = 32;
pub const LAYERS: usize = 4;
pub const H_KV: usize = 1;
pub const HD: usize = 32;
const INTER: usize = 16;
const H_K: usize = 2;
const H_V: usize = 4;
const S: usize = 4;
const KERNEL: usize = 4;
const H: usize = 2;
const ROPE_DIM: usize = 16;
const ROT_BLOCK: usize = 4;

pub struct Lcg(pub u64);

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

    fn tensor<R: Runtime<DType = DType>>(
        &mut self,
        device: &R::Device,
        shape: &[usize],
        scale: f32,
    ) -> Tensor<R> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.uniform(-scale, scale)).collect();
        Tensor::<R>::from_slice(&data, shape, device).unwrap()
    }
}

/// Four layers, attention at odd indices (`full_attention_interval = 2`).
pub fn tiny_config() -> UniversalConfig {
    UniversalConfig {
        model_type: "qwen35".into(),
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_layers: LAYERS,
        max_seq_len: MAX_POS,
        intermediate_size: Some(INTER),
        rms_norm_eps: 1e-6,
        attention: None,
        ssm: None,
        moe: None,
        hybrid_layers: Some(HybridConfig::from_full_attention_interval(LAYERS, 2)),
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
            rope_sections: [3, 3, 2, 0],
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }),
        hadamard: None,
        quant_formats: Vec::new(),
    }
}

struct Builder<'a, R: Runtime<DType = DType>> {
    device: &'a R::Device,
    rng: Lcg,
    rotations: HashMap<usize, HadamardRotation<R>>,
}

impl<R: Runtime<DType = DType>> Builder<'_, R>
where
    R::Client: ShapeOps<R>,
{
    fn rotation(&mut self, inp: usize) -> HadamardRotation<R> {
        if let Some(r) = self.rotations.get(&inp) {
            return r.clone();
        }
        let signs: Vec<i8> = (0..inp).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let rotation =
            HadamardRotation::<R>::new(ROT_BLOCK, Some(&signs), DType::F32, self.device).unwrap();
        self.rotations.insert(inp, rotation.clone());
        rotation
    }

    fn plain(&mut self, out: usize, inp: usize) -> MaybeQuantLinear<R> {
        let w = self
            .rng
            .tensor::<R>(self.device, &[out, inp], 0.5 / (inp as f32).sqrt());
        MaybeQuantLinear::Standard(Linear::new(w, None, false))
    }

    fn linear(&mut self, out: usize, inp: usize) -> MaybeRotatedLinear<R> {
        let base = self.plain(out, inp);
        let rotation = self.rotation(inp);
        MaybeRotatedLinear::Rotated(Box::new(RotatedLinear::new(base, rotation).unwrap()))
    }

    fn norm(&mut self) -> RmsNorm<R> {
        let w: Vec<f32> = (0..HIDDEN).map(|_| self.rng.uniform(0.5, 1.5)).collect();
        let w = Tensor::<R>::from_slice(&w, &[HIDDEN], self.device).unwrap();
        RmsNorm::new(w, 1e-6, false)
    }

    fn mlp(&mut self) -> RotatedMlp<R> {
        let gate = self.linear(INTER, HIDDEN);
        let up = self.linear(INTER, HIDDEN);
        let down = self.linear(HIDDEN, INTER);
        RotatedMlp::new(gate, up, down).unwrap()
    }

    fn gdn(&mut self, cfg: &GdnConfig) -> GdnBlock<R> {
        let a: Vec<f32> = (0..H_V)
            .map(|_| -self.rng.uniform(0.5, 1.5).exp())
            .collect();
        let weights = GdnWeights {
            attn_qkv: self.linear(cfg.qkv_dim(), HIDDEN),
            attn_gate: self.linear(cfg.value_dim(), HIDDEN),
            ssm_alpha: self.plain(H_V, HIDDEN),
            ssm_beta: self.plain(H_V, HIDDEN),
            ssm_out: self.linear(HIDDEN, cfg.value_dim()),
            ssm_conv1d: self
                .rng
                .tensor::<R>(self.device, &[cfg.qkv_dim(), KERNEL], 0.5),
            ssm_a: Tensor::<R>::from_slice(&a, &[H_V], self.device).unwrap(),
            ssm_dt_bias: self.rng.tensor::<R>(self.device, &[H_V], 0.5),
            ssm_norm: self.rng.tensor::<R>(self.device, &[S], 1.0),
        };
        GdnBlock::new(cfg.clone(), weights).unwrap()
    }

    fn attention(&mut self, cfg: &Qwen35AttentionConfig) -> Qwen35AttentionBlock<R> {
        let q_norm: Vec<f32> = (0..HD).map(|_| self.rng.uniform(0.5, 1.5)).collect();
        let weights = Qwen35AttentionWeights {
            attn_q: self.linear(cfg.q_gate_dim(), HIDDEN),
            attn_k: self.linear(cfg.kv_dim(), HIDDEN),
            attn_v: self.linear(cfg.kv_dim(), HIDDEN),
            attn_output: self.linear(HIDDEN, cfg.q_dim()),
            attn_q_norm: Tensor::<R>::from_slice(&q_norm, &[HD], self.device).unwrap(),
            attn_k_norm: self.rng.tensor::<R>(self.device, &[HD], 1.0),
        };
        Qwen35AttentionBlock::new(cfg.clone(), weights).unwrap()
    }
}

/// Build the tiny model with weights drawn from `seed`.
pub fn tiny_model<R: Runtime<DType = DType>>(device: &R::Device, seed: u64) -> Qwen35Model<R>
where
    R::Client: ShapeOps<R>,
{
    let config = tiny_config();
    let gdn_cfg = config.gdn.clone().unwrap();
    let attn_cfg = config.qwen35_attention.clone().unwrap();
    let layers = config.hybrid_layers.clone().unwrap();
    let mut b = Builder::<R> {
        device,
        rng: Lcg(seed),
        rotations: HashMap::new(),
    };
    let table = b.rng.tensor::<R>(device, &[VOCAB, HIDDEN], 1.0);
    let embed =
        MaybeRotatedEmbedding::Plain(MaybeQuantEmbedding::Standard(Embedding::new(table, false)));
    let mut blocks = Vec::new();
    for i in 0..LAYERS {
        let attn_norm = b.norm();
        let post_attention_norm = b.norm();
        let mlp = b.mlp();
        if layers.is_attention_layer(i) {
            blocks.push(Qwen35Block::Attention(Box::new(Qwen35AttentionLayer {
                attn_norm,
                mixer: b.attention(&attn_cfg),
                post_attention_norm,
                mlp,
            })));
        } else {
            blocks.push(Qwen35Block::Gdn(Box::new(Qwen35GdnLayer {
                attn_norm,
                mixer: b.gdn(&gdn_cfg),
                post_attention_norm,
                mlp,
            })));
        }
    }
    let norm = b.norm();
    let lm_head = b.linear(VOCAB, HIDDEN);
    let rope = attn_cfg.rope_table::<R>(MAX_POS, device).unwrap();
    Qwen35Model::new(config, embed, blocks, norm, lm_head, rope).unwrap()
}

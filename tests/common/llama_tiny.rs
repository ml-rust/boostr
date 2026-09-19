//! A tiny synthetic Llama on the CUDA runtime, loaded through the same
//! `VarMap` -> `VarBuilder` -> `LoadedModel::load` path a checkpoint takes.
//! Deterministic weights from a seed; every weight is cast to one dtype so
//! the model's activations, its RoPE tables and its KV cache share it.
//!
//! Head dim is 32, the smallest the CUDA graph decode kernel accepts.

use boostr::model::LoadedModel;
use boostr::model::config::{AttentionConfig, ModelConfig};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

pub const VOCAB: usize = 64;
pub const HIDDEN: usize = 128;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 32;
pub const INTER: usize = 256;
pub const MAX_POS: usize = 4096;

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
}

pub fn tiny_config() -> ModelConfig {
    ModelConfig {
        model_type: "llama".into(),
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_layers: LAYERS,
        max_seq_len: MAX_POS,
        intermediate_size: Some(INTER),
        rms_norm_eps: 1e-5,
        attention: Some(AttentionConfig {
            num_heads: HEADS,
            num_kv_heads: Some(KV_HEADS),
            head_dim: Some(HEAD_DIM),
            rope_theta: 10_000.0,
            rope_scaling: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            sliding_window: None,
            use_alibi: false,
        }),
        ssm: None,
        moe: None,
        hybrid_layers: None,
        tie_word_embeddings: false,
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: None,
        qwen35_attention: None,
        hadamard: None,
        quant_formats: Vec::new(),
    }
}

struct Weights<'a> {
    device: &'a CudaDevice,
    rng: Lcg,
    dtype: DType,
    map: VarMap<CudaRuntime>,
}

impl Weights<'_> {
    fn put(&mut self, name: &str, shape: &[usize], lo: f32, hi: f32) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.rng.uniform(lo, hi)).collect();
        let t = Tensor::<CudaRuntime>::from_slice(&data, shape, self.device).unwrap();
        let t = if self.dtype == DType::F32 {
            t
        } else {
            let client = CudaRuntime::default_client(self.device);
            client.cast(&t, self.dtype).unwrap()
        };
        self.map.insert(name.to_string(), t);
    }

    fn linear(&mut self, name: &str, out: usize, inp: usize) {
        let scale = 0.5 / (inp as f32).sqrt();
        self.put(name, &[out, inp], -scale, scale);
    }

    fn norm(&mut self, name: &str) {
        self.put(name, &[HIDDEN], 0.5, 1.5);
    }
}

/// Build the model at `dtype` from `seed`.
pub fn tiny_llama(device: &CudaDevice, seed: u64, dtype: DType) -> LoadedModel<CudaRuntime> {
    let mut w = Weights {
        device,
        rng: Lcg(seed),
        dtype,
        map: VarMap::new(),
    };
    w.put("model.embed_tokens.weight", &[VOCAB, HIDDEN], -0.5, 0.5);
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        w.norm(&format!("{p}.input_layernorm.weight"));
        w.linear(
            &format!("{p}.self_attn.q_proj.weight"),
            HEADS * HEAD_DIM,
            HIDDEN,
        );
        w.linear(
            &format!("{p}.self_attn.k_proj.weight"),
            KV_HEADS * HEAD_DIM,
            HIDDEN,
        );
        w.linear(
            &format!("{p}.self_attn.v_proj.weight"),
            KV_HEADS * HEAD_DIM,
            HIDDEN,
        );
        w.linear(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            HEADS * HEAD_DIM,
        );
        w.norm(&format!("{p}.post_attention_layernorm.weight"));
        w.linear(&format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN);
        w.linear(&format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN);
        w.linear(&format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER);
    }
    w.norm("model.norm.weight");
    w.linear("lm_head.weight", VOCAB, HIDDEN);

    let config = tiny_config();
    let mut map = w.map;
    let mut vb = VarBuilder::new(&mut map, device);
    LoadedModel::<CudaRuntime>::load(&config, &mut vb).unwrap()
}

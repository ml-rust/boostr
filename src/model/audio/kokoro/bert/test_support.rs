//! Shared test-construction helpers for the `bert` module's test suites.

use super::embeddings::{AlbertConfig, AlbertEmbeddings};
use super::layer::AlbertLayer;
use crate::nn::Embedding;
use numr::runtime::{Runtime, cpu::CpuRuntime};
use numr::tensor::Tensor;

pub(crate) fn zeros(
    shape: &[usize],
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device).unwrap()
}
pub(crate) fn ones(
    shape: &[usize],
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device).unwrap()
}

pub(crate) fn tiny_config() -> AlbertConfig {
    AlbertConfig {
        hidden_size: 4,
        embedding_size: 2,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 8,
        max_position_embeddings: 16,
        vocab_size: 10,
        type_vocab_size: 2,
        layer_norm_eps: 1e-5,
    }
}

pub(crate) fn build_layer(
    cfg: &AlbertConfig,
    device: &<CpuRuntime as Runtime>::Device,
) -> AlbertLayer<CpuRuntime> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    AlbertLayer {
        q_weight: zeros(&[h, h], device),
        q_bias: zeros(&[h], device),
        k_weight: zeros(&[h, h], device),
        k_bias: zeros(&[h], device),
        v_weight: zeros(&[h, h], device),
        v_bias: zeros(&[h], device),
        attn_dense_weight: zeros(&[h, h], device),
        attn_dense_bias: zeros(&[h], device),
        attn_ln_weight: ones(&[h], device),
        attn_ln_bias: zeros(&[h], device),
        ffn_weight: zeros(&[i, h], device),
        ffn_bias: zeros(&[i], device),
        ffn_output_weight: zeros(&[h, i], device),
        ffn_output_bias: zeros(&[h], device),
        full_ln_weight: ones(&[h], device),
        full_ln_bias: zeros(&[h], device),
    }
}

pub(crate) fn build_embeddings(
    cfg: &AlbertConfig,
    device: &<CpuRuntime as Runtime>::Device,
) -> AlbertEmbeddings<CpuRuntime> {
    AlbertEmbeddings::new(
        Embedding::new(zeros(&[cfg.vocab_size, cfg.embedding_size], device), false),
        Embedding::new(
            zeros(&[cfg.max_position_embeddings, cfg.embedding_size], device),
            false,
        ),
        Embedding::new(
            zeros(&[cfg.type_vocab_size, cfg.embedding_size], device),
            false,
        ),
        ones(&[cfg.embedding_size], device),
        zeros(&[cfg.embedding_size], device),
        cfg.layer_norm_eps,
        cfg.max_position_embeddings,
    )
}

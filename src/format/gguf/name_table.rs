//! The GGUF <-> HuggingFace tensor name pairs, in one table for both directions.
//!
//! [`gguf_to_hf_name`](super::gguf_to_hf_name) reads a GGUF file into the names
//! `LoadedModel::load()` expects. [`hf_to_gguf_name`](super::hf_to_gguf_name)
//! writes a HuggingFace checkpoint under llama.cpp's names. Both read these
//! tables, so the two directions cannot drift apart.

/// Whole-name pairs for tensors outside the layer stack: `(gguf, hf)`.
pub(crate) const GLOBAL_NAMES: &[(&str, &str)] = &[
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
];

/// Layer-suffix pairs: `blk.N.<gguf>` <-> `model.layers.N.<hf>`.
pub(crate) const LAYER_SUFFIXES: &[(&str, &str)] = &[
    // Standard attention
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_output.weight", "self_attn.o_proj.weight"),
    // Attention biases
    ("attn_q.bias", "self_attn.q_proj.bias"),
    ("attn_k.bias", "self_attn.k_proj.bias"),
    ("attn_v.bias", "self_attn.v_proj.bias"),
    ("attn_output.bias", "self_attn.o_proj.bias"),
    // Per-head query and key norms (Qwen3, OLMo2, Gemma3)
    ("attn_q_norm.weight", "self_attn.q_norm.weight"),
    ("attn_k_norm.weight", "self_attn.k_norm.weight"),
    // MLA (Multi-Head Latent Attention) - DeepSeek-V2/V3
    ("attn_q_a.weight", "self_attn.q_a_proj.weight"),
    ("attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight"),
    ("attn_q_b.weight", "self_attn.q_b_proj.weight"),
    ("attn_kv_a.weight", "self_attn.kv_a_proj_with_mqa.weight"),
    ("attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"),
    ("attn_kv_b.weight", "self_attn.kv_b_proj.weight"),
    // Standard MLP (SwiGLU)
    ("ffn_gate.weight", "mlp.gate_proj.weight"),
    ("ffn_up.weight", "mlp.up_proj.weight"),
    ("ffn_down.weight", "mlp.down_proj.weight"),
    // MoE router
    ("ffn_gate_inp.weight", "block_sparse_moe.gate.weight"),
    ("ffn_gate_inp.bias", "block_sparse_moe.gate.bias"),
    // MoE shared expert
    (
        "ffn_gate_shexp.weight",
        "block_sparse_moe.shared_expert.gate_proj.weight",
    ),
    (
        "ffn_up_shexp.weight",
        "block_sparse_moe.shared_expert.up_proj.weight",
    ),
    (
        "ffn_down_shexp.weight",
        "block_sparse_moe.shared_expert.down_proj.weight",
    ),
    // Layer norms
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    // Mamba2 SSM
    ("ssm_in.weight", "mamba2.mixer.in_proj.weight"),
    ("ssm_in.bias", "mamba2.mixer.in_proj.bias"),
    ("ssm_out.weight", "mamba2.mixer.out_proj.weight"),
    ("ssm_out.bias", "mamba2.mixer.out_proj.bias"),
    ("ssm_conv1d.weight", "mamba2.mixer.conv1d.weight"),
    ("ssm_conv1d.bias", "mamba2.mixer.conv1d.bias"),
    ("ssm_a", "mamba2.mixer.A_log"),
    ("ssm_dt.bias", "mamba2.mixer.dt_bias"),
    ("ssm_d", "mamba2.mixer.D"),
    ("ssm_norm.weight", "mamba2.mixer.norm.weight"),
];

/// The HF side of a GGUF entry in `table`.
pub(crate) fn hf_for(table: &[(&'static str, &'static str)], gguf: &str) -> Option<&'static str> {
    table.iter().find(|(g, _)| *g == gguf).map(|(_, hf)| *hf)
}

/// The GGUF side of an HF entry in `table`.
pub(crate) fn gguf_for(table: &[(&'static str, &'static str)], hf: &str) -> Option<&'static str> {
    table.iter().find(|(_, h)| *h == hf).map(|(g, _)| *g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A duplicated side would make one direction ambiguous.
    #[test]
    fn each_side_of_each_table_is_unique() {
        for table in [GLOBAL_NAMES, LAYER_SUFFIXES] {
            for (i, (g, h)) in table.iter().enumerate() {
                for (g2, h2) in &table[i + 1..] {
                    assert_ne!(g, g2, "GGUF name {g} appears twice");
                    assert_ne!(h, h2, "HF name {h} appears twice");
                }
            }
        }
    }
}

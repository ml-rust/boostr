//! StarCoder2, Mixtral and Qwen2-MoE: Llama prefixes with their own MLP names.

/// StarCoder2 MLP naming:
/// - `mlp.c_fc.*` → `mlp.up_proj.*`
/// - `mlp.c_proj.*` → `mlp.down_proj.*`
pub(super) fn normalize_starcoder2(name: &str) -> String {
    name.replace(".mlp.c_fc.", ".mlp.up_proj.")
        .replace(".mlp.c_proj.", ".mlp.down_proj.")
}

/// Mixtral expert naming:
/// - `block_sparse_moe.experts.{E}.w1.*` → `...experts.{E}.gate_proj.*`
/// - `block_sparse_moe.experts.{E}.w3.*` → `...experts.{E}.up_proj.*`
/// - `block_sparse_moe.experts.{E}.w2.*` → `...experts.{E}.down_proj.*`
pub(super) fn normalize_mixtral(name: &str) -> String {
    let Some(pos) = name.find(".block_sparse_moe.experts.") else {
        return name.to_string();
    };
    let (head, tail) = name.split_at(pos);
    let tail = tail
        .replace(".w1.", ".gate_proj.")
        .replace(".w3.", ".up_proj.")
        .replace(".w2.", ".down_proj.");
    format!("{head}{tail}")
}

/// Qwen2-MoE naming: a sparse layer's `mlp.{gate, experts, shared_expert,
/// shared_expert_gate}` → `block_sparse_moe.*`. A dense layer keeps
/// `mlp.{gate,up,down}_proj`.
pub(super) fn normalize_qwen2_moe(name: &str) -> String {
    for part in ["gate.", "experts.", "shared_expert.", "shared_expert_gate."] {
        let from = format!(".mlp.{part}");
        if name.contains(&from) {
            return name.replace(&from, &format!(".block_sparse_moe.{part}"));
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::normalize_hf_name;

    #[test]
    fn starcoder2_mlp() {
        assert_eq!(
            normalize_hf_name("starcoder2", "model.layers.3.mlp.c_fc.bias"),
            "model.layers.3.mlp.up_proj.bias"
        );
        assert_eq!(
            normalize_hf_name("starcoder2", "model.layers.3.mlp.c_proj.weight"),
            "model.layers.3.mlp.down_proj.weight"
        );
    }

    #[test]
    fn mixtral_experts() {
        for (from, to) in [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")] {
            assert_eq!(
                normalize_hf_name(
                    "mixtral",
                    &format!("model.layers.0.block_sparse_moe.experts.7.{from}.weight")
                ),
                format!("model.layers.0.block_sparse_moe.experts.7.{to}.weight")
            );
        }
        assert_eq!(
            normalize_hf_name("mixtral", "model.layers.0.block_sparse_moe.gate.weight"),
            "model.layers.0.block_sparse_moe.gate.weight"
        );
    }

    #[test]
    fn qwen2_moe_sparse_and_dense() {
        assert_eq!(
            normalize_hf_name("qwen2_moe", "model.layers.1.mlp.shared_expert_gate.weight"),
            "model.layers.1.block_sparse_moe.shared_expert_gate.weight"
        );
        assert_eq!(
            normalize_hf_name("qwen2_moe", "model.layers.1.mlp.experts.3.up_proj.weight"),
            "model.layers.1.block_sparse_moe.experts.3.up_proj.weight"
        );
        assert_eq!(
            normalize_hf_name("qwen2_moe", "model.layers.1.mlp.gate.weight"),
            "model.layers.1.block_sparse_moe.gate.weight"
        );
        assert_eq!(
            normalize_hf_name("qwen2_moe", "model.layers.1.mlp.gate_proj.weight"),
            "model.layers.1.mlp.gate_proj.weight"
        );
    }
}

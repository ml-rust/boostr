//! Per-architecture parameter detectors (shape-based inference).

use super::types::DetectedConfig;
use crate::format::SafeTensors;

/// Detect Mamba2 parameters from tensor shapes
pub(super) fn detect_mamba2_params(
    safetensors: &SafeTensors,
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.mamba2.mixer.", model_prefix, layer_idx);

    // Detect num_heads from A_log shape
    if let Ok(info) = safetensors.tensor_info(&format!("{}A_log", prefix)) {
        if info.shape.len() == 1 {
            config.mamba2_num_heads = Some(info.shape[0]);
        }
    }

    // Detect conv_kernel from conv1d weight shape [d_inner, 1, kernel]
    if let Ok(info) = safetensors.tensor_info(&format!("{}conv1d.weight", prefix)) {
        if info.shape.len() == 3 {
            config.mamba2_conv_kernel = Some(info.shape[2]);
            let d_inner = info.shape[0];
            if config.hidden_size > 0 {
                config.mamba2_expand = Some(d_inner / config.hidden_size);
            }
        }
    }

    // Detect head_dim from norm weight (d_inner = num_heads * head_dim)
    if let Ok(info) = safetensors.tensor_info(&format!("{}norm.weight", prefix)) {
        if info.shape.len() == 1 {
            let d_inner = info.shape[0];
            if let Some(num_heads) = config.mamba2_num_heads {
                config.mamba2_head_dim = Some(d_inner / num_heads);
            }
        }
    }

    // Detect state_size from in_proj dimensions
    // out_dim = d_inner (z) + d_inner + 2*state_size (xBC) + num_heads (dt)
    // => state_size = (out_dim - 2*d_inner - num_heads) / 2
    if let Ok(info) = safetensors.tensor_info(&format!("{}in_proj.weight", prefix)) {
        if info.shape.len() == 2 {
            let out_dim = info.shape[0];
            if let (Some(num_heads), Some(expand)) = (config.mamba2_num_heads, config.mamba2_expand)
            {
                let d_inner = config.hidden_size * expand;
                let state_size = (out_dim - 2 * d_inner - num_heads) / 2;
                config.mamba2_state_size = Some(state_size);
            }
        }
    }
}

/// Detect Mamba3 parameters from tensor shapes
pub(super) fn detect_mamba3_params(
    safetensors: &SafeTensors,
    tensor_names: &[&str],
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.mamba3.mixer.", model_prefix, layer_idx);

    config.mamba3_enabled = Some(true);

    // Detect complex RoPE from theta_proj presence
    let has_theta_proj = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}theta_proj", prefix)));
    config.mamba3_complex_rope = Some(has_theta_proj);

    // Detect MIMO from mimo_x_up presence
    let has_mimo = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}mimo_x_up", prefix)));
    if has_mimo {
        if let Ok(info) = safetensors.tensor_info(&format!("{}mimo_x_up.weight", prefix)) {
            if info.shape.len() == 2 {
                // mimo_x_up: [head_dim * mimo_rank, head_dim]
                let out_dim = info.shape[0];
                let in_dim = info.shape[1];
                if in_dim > 0 {
                    config.mamba3_mimo_rank = Some(out_dim / in_dim);
                }
            }
        }
    } else {
        config.mamba3_mimo_rank = Some(0); // SISO mode
    }

    // Detect conv usage from conv1d presence
    let has_conv = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}conv1d", prefix)));
    config.mamba3_use_conv = Some(has_conv);
}

/// Detect MLA parameters from tensor shapes
pub(super) fn detect_mla_params(
    safetensors: &SafeTensors,
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.self_attn.", model_prefix, layer_idx);

    // Detect kv_latent_dim from w_dkv weight shape [kv_latent_dim, hidden_size]
    if let Ok(info) = safetensors.tensor_info(&format!("{}w_dkv.weight", prefix)) {
        if info.shape.len() == 2 {
            config.kv_latent_dim = Some(info.shape[0]);
        }
    }

    // Detect q_latent_dim from w_dq weight shape [q_latent_dim, hidden_size]
    if let Ok(info) = safetensors.tensor_info(&format!("{}w_dq.weight", prefix)) {
        if info.shape.len() == 2 {
            config.q_latent_dim = Some(info.shape[0]);
        }
    }

    // Detect d_rope from w_kr weight shape [d_rope, hidden_size]
    if let Ok(info) = safetensors.tensor_info(&format!("{}w_kr.weight", prefix)) {
        if info.shape.len() == 2 {
            config.d_rope = Some(info.shape[0]);
        }
    }

    // Detect num_attention_heads from w_qr weight shape [d_rope * num_heads, hidden_size]
    if let Ok(info) = safetensors.tensor_info(&format!("{}w_qr.weight", prefix)) {
        if info.shape.len() == 2 {
            if let Some(d_rope) = config.d_rope {
                config.num_attention_heads = Some(info.shape[0] / d_rope);
            }
        }
    }
}

/// Detect MoE parameters from tensor shapes
pub(super) fn detect_moe_params(
    safetensors: &SafeTensors,
    tensor_names: &[&str],
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.moe.", model_prefix, layer_idx);

    // Count experts
    let mut max_expert_idx = 0;
    for name in tensor_names {
        if let Some(rest) = name.strip_prefix(&format!("{}experts.", prefix)) {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                    max_expert_idx = max_expert_idx.max(idx + 1);
                }
            }
        }
    }
    if max_expert_idx > 0 {
        config.num_experts = Some(max_expert_idx);
    }

    // Detect intermediate_size from expert gate_proj weight shape
    if let Ok(info) = safetensors.tensor_info(&format!("{}experts.0.gate_proj.weight", prefix)) {
        if info.shape.len() == 2 {
            config.intermediate_size = Some(info.shape[0]);
        }
    }

    // Check for shared expert
    config.shared_expert_enabled = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}shared_expert.", prefix)));
}

/// Detect standard transformer parameters
pub(super) fn detect_transformer_params(
    safetensors: &SafeTensors,
    _tensor_names: &[&str],
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.self_attn.", model_prefix, layer_idx);

    // Detect num_attention_heads from q_proj weight shape
    if let Ok(info) = safetensors.tensor_info(&format!("{}q_proj.weight", prefix)) {
        if info.shape.len() == 2 && config.hidden_size > 0 {
            let out_dim = info.shape[0];
            for head_dim in [64, 128, 96, 80] {
                if out_dim % head_dim == 0 {
                    config.num_attention_heads = Some(out_dim / head_dim);
                    config.head_dim = Some(head_dim);
                    break;
                }
            }
            if config.num_attention_heads.is_none() {
                config.num_attention_heads = Some(out_dim / (config.hidden_size / 8).max(1));
            }
        }
    }

    // Detect num_kv_heads from k_proj weight shape
    if let Ok(info) = safetensors.tensor_info(&format!("{}k_proj.weight", prefix)) {
        if info.shape.len() == 2 && config.hidden_size > 0 {
            let kv_out_dim = info.shape[0];
            if let Some(head_dim) = config.head_dim {
                config.num_kv_heads = Some(kv_out_dim / head_dim);
            } else if let Some(num_heads) = config.num_attention_heads {
                let head_dim = config.hidden_size / num_heads;
                config.num_kv_heads = Some(kv_out_dim / head_dim);
            }
        }
    }

    // Detect intermediate_size from mlp gate_proj
    let mlp_prefix = format!("{}layers.{}.mlp.", model_prefix, layer_idx);
    if let Ok(info) = safetensors.tensor_info(&format!("{}gate_proj.weight", mlp_prefix)) {
        if info.shape.len() == 2 {
            config.intermediate_size = Some(info.shape[0]);
        }
    }
}

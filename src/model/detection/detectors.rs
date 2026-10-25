//! Per-architecture parameter detectors (shape-based inference).
//!
//! Each detector reads shapes through a [`ShapeOf`] lookup, so a single-file
//! [`SafeTensors`](crate::format::SafeTensors) and a sharded loader feed the
//! same code.

use super::types::DetectedConfig;

/// The shape of the tensor a full name names, or `None` when it is absent.
pub type ShapeOf<'a> = &'a dyn Fn(&str) -> Option<Vec<usize>>;

/// Detect Mamba2 parameters from tensor shapes.
///
/// Reads the mixer under `mamba2.` (oxidizr) or `mamba2.mixer.`. The inner
/// width comes from `out_proj` `[hidden, d_inner]`, and the state size from
/// `in_proj`, whose rows are `2 * d_inner + 2 * state + num_heads` with one
/// group.
pub(super) fn detect_mamba2_params(
    shape_of: ShapeOf<'_>,
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let base = format!("{}layers.{}.mamba2.", model_prefix, layer_idx);
    let nested = format!("{base}mixer.");
    let prefix = if shape_of(&format!("{nested}A_log")).is_some() {
        nested
    } else {
        base
    };
    let shape = |name: &str| shape_of(&format!("{prefix}{name}"));

    if let Some([num_heads]) = shape("A_log").as_deref() {
        config.mamba2_num_heads = Some(*num_heads);
    }
    if let Some([_, _, kernel]) = shape("conv1d.weight").as_deref() {
        config.mamba2_conv_kernel = Some(*kernel);
    }
    let d_inner = match shape("out_proj.weight").as_deref() {
        Some([_, d_inner]) => Some(*d_inner),
        _ => match shape("norm.weight").as_deref() {
            Some([d_inner]) => Some(*d_inner),
            _ => None,
        },
    };
    let Some(d_inner) = d_inner else {
        return;
    };
    if config.hidden_size > 0 && d_inner.is_multiple_of(config.hidden_size) {
        config.mamba2_expand = Some(d_inner / config.hidden_size);
    }
    let Some(num_heads) = config.mamba2_num_heads.filter(|&h| h > 0) else {
        return;
    };
    config.mamba2_head_dim = Some(d_inner / num_heads);
    if let Some([rows, _]) = shape("in_proj.weight").as_deref()
        && let Some(rest) = rows.checked_sub(2 * d_inner + num_heads)
    {
        config.mamba2_state_size = Some(rest / 2);
    }
}

/// Detect Mamba3 parameters from tensor shapes
pub(super) fn detect_mamba3_params(
    shape_of: ShapeOf<'_>,
    tensor_names: &[&str],
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.mamba3.mixer.", model_prefix, layer_idx);
    let has = |stem: &str| {
        tensor_names
            .iter()
            .any(|k| k.starts_with(&format!("{prefix}{stem}")))
    };

    config.mamba3_enabled = Some(true);
    // Complex RoPE from theta_proj presence
    config.mamba3_complex_rope = Some(has("theta_proj"));
    // MIMO from mimo_x_up: [head_dim * mimo_rank, head_dim]. SISO is rank 0.
    config.mamba3_mimo_rank = if has("mimo_x_up") {
        match shape_of(&format!("{prefix}mimo_x_up.weight")).as_deref() {
            Some([out_dim, in_dim]) => out_dim.checked_div(*in_dim),
            _ => None,
        }
    } else {
        Some(0)
    };
    config.mamba3_use_conv = Some(has("conv1d"));
}

/// Detect MLA parameters from tensor shapes, by the names oxidizr writes:
///
/// - `kv_a_layernorm` `[kv_latent_dim]`
/// - `kv_a_proj_with_mqa` `[kv_latent_dim + d_rope, hidden]`
/// - `q_a_proj` `[q_latent_dim, hidden]`, absent without Q compression
/// - `q_b_proj` (or `q_proj`) `[num_heads * (hidden / num_heads + d_rope), ..]`,
///   so `num_heads = (rows - hidden) / d_rope`
/// - `mlp.gate_proj` `[intermediate, hidden]` when the layer has an MLP
pub(super) fn detect_mla_params(
    shape_of: ShapeOf<'_>,
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let layer = format!("{}layers.{}.", model_prefix, layer_idx);
    let rows =
        |name: &str| shape_of(&format!("{layer}{name}")).and_then(|shape| shape.first().copied());

    if let Some(intermediate) = rows("mlp.gate_proj.weight") {
        config.intermediate_size = Some(intermediate);
    }
    let Some(kv_latent) = rows("self_attn.kv_a_layernorm.weight") else {
        return;
    };
    config.kv_latent_dim = Some(kv_latent);
    if let Some(kv_a) = rows("self_attn.kv_a_proj_with_mqa.weight") {
        config.d_rope = kv_a.checked_sub(kv_latent).filter(|&d| d > 0);
    }
    config.q_latent_dim = rows("self_attn.q_a_proj.weight");
    let q_rows = rows("self_attn.q_b_proj.weight").or_else(|| rows("self_attn.q_proj.weight"));
    if let (Some(q_rows), Some(d_rope)) = (q_rows, config.d_rope)
        && let Some(rope_rows) = q_rows.checked_sub(config.hidden_size)
        && rope_rows.is_multiple_of(d_rope)
    {
        config.num_attention_heads = Some(rope_rows / d_rope);
    }
}

/// Detect MoE parameters from tensor shapes
pub(super) fn detect_moe_params(
    shape_of: ShapeOf<'_>,
    tensor_names: &[&str],
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.moe.", model_prefix, layer_idx);

    // Count experts
    let mut max_expert_idx = 0;
    for name in tensor_names {
        if let Some(rest) = name.strip_prefix(&format!("{}experts.", prefix))
            && let Some(dot_pos) = rest.find('.')
            && let Ok(idx) = rest[..dot_pos].parse::<usize>()
        {
            max_expert_idx = max_expert_idx.max(idx + 1);
        }
    }
    if max_expert_idx > 0 {
        config.num_experts = Some(max_expert_idx);
    }

    // Detect intermediate_size from expert gate_proj weight shape
    if let Some([intermediate, _]) =
        shape_of(&format!("{}experts.0.gate_proj.weight", prefix)).as_deref()
    {
        config.intermediate_size = Some(*intermediate);
    }

    // Check for shared expert
    config.shared_expert_enabled = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}shared_expert.", prefix)));
}

/// Detect standard transformer parameters
pub(super) fn detect_transformer_params(
    shape_of: ShapeOf<'_>,
    layer_idx: usize,
    config: &mut DetectedConfig,
    model_prefix: &str,
) {
    let prefix = format!("{}layers.{}.self_attn.", model_prefix, layer_idx);

    // Detect num_attention_heads from q_proj weight shape
    if let Some([out_dim, _]) = shape_of(&format!("{}q_proj.weight", prefix)).as_deref()
        && config.hidden_size > 0
    {
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

    // Detect num_kv_heads from k_proj weight shape
    if let Some([kv_out_dim, _]) = shape_of(&format!("{}k_proj.weight", prefix)).as_deref()
        && config.hidden_size > 0
    {
        if let Some(head_dim) = config.head_dim {
            config.num_kv_heads = Some(kv_out_dim / head_dim);
        } else if let Some(num_heads) = config.num_attention_heads {
            let head_dim = config.hidden_size / num_heads;
            config.num_kv_heads = Some(kv_out_dim / head_dim);
        }
    }

    // Detect intermediate_size from mlp gate_proj
    let mlp_prefix = format!("{}layers.{}.mlp.", model_prefix, layer_idx);
    if let Some([intermediate, _]) = shape_of(&format!("{}gate_proj.weight", mlp_prefix)).as_deref()
    {
        config.intermediate_size = Some(*intermediate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(shapes: &[(&str, &[usize])]) -> HashMap<String, Vec<usize>> {
        shapes
            .iter()
            .map(|(n, s)| (n.to_string(), s.to_vec()))
            .collect()
    }

    /// nano-start's MLA layer: hidden 128, 4 heads, KV rank 64, Q rank 64,
    /// rotary width 8.
    #[test]
    fn mla_dims_come_from_the_oxidizr_shapes() {
        let shapes = lookup(&[
            ("model.layers.3.self_attn.q_a_proj.weight", &[64, 128]),
            ("model.layers.3.self_attn.q_b_proj.weight", &[160, 64]),
            (
                "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
                &[72, 128],
            ),
            ("model.layers.3.self_attn.kv_a_layernorm.weight", &[64]),
        ]);
        let mut config = DetectedConfig {
            hidden_size: 128,
            ..DetectedConfig::default()
        };
        detect_mla_params(&|n| shapes.get(n).cloned(), 3, &mut config, "model.");
        assert_eq!(config.kv_latent_dim, Some(64));
        assert_eq!(config.q_latent_dim, Some(64));
        assert_eq!(config.d_rope, Some(8));
        assert_eq!(config.num_attention_heads, Some(4));
    }

    /// nano-start's Mamba2 layer: 16 heads of 16, state 32, kernel 4.
    #[test]
    fn mamba2_dims_come_from_the_oxidizr_shapes() {
        let shapes = lookup(&[
            ("model.layers.0.mamba2.A_log", &[16]),
            ("model.layers.0.mamba2.conv1d.weight", &[320, 1, 4]),
            ("model.layers.0.mamba2.in_proj.weight", &[592, 128]),
            ("model.layers.0.mamba2.out_proj.weight", &[128, 256]),
        ]);
        let mut config = DetectedConfig {
            hidden_size: 128,
            ..DetectedConfig::default()
        };
        detect_mamba2_params(&|n| shapes.get(n).cloned(), 0, &mut config, "model.");
        assert_eq!(config.mamba2_num_heads, Some(16));
        assert_eq!(config.mamba2_head_dim, Some(16));
        assert_eq!(config.mamba2_expand, Some(2));
        assert_eq!(config.mamba2_state_size, Some(32));
        assert_eq!(config.mamba2_conv_kernel, Some(4));
    }
}

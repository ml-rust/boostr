//! [`mla_refusal`]: why a Llama refuses an MLA config.

use crate::model::config::UniversalConfig;

/// Why the Llama model cannot run `config`, or `None` when its attention is
/// not MLA.
///
/// `attention.kv_latent_dim` selects MLA, whose layers hold `q_a_proj`,
/// `kv_a_proj_with_mqa` and `kv_b_proj` in place of `q_proj`, `k_proj` and
/// `v_proj`. [`LoadedModel::load`](crate::model::LoadedModel::load) builds
/// an MLA config as the hybrid model.
pub fn mla_refusal(config: &UniversalConfig) -> Option<String> {
    let attn = config.attention.as_ref()?;
    attn.is_mla().then(|| {
        format!(
            "model type '{}' sets attention.kv_latent_dim, which selects MLA attention; the \
             Llama model runs dense attention only. Load it through LoadedModel::load, which \
             builds MLA as the hybrid model",
            config.model_type
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_configs_are_refused_and_dense_ones_pass() {
        let yaml = "model_type: llama\nvocab_size: 16\nhidden_size: 16\nnum_layers: 1\n\
                    max_seq_len: 8\nintermediate_size: 32\nattention:\n  num_heads: 2\n";
        let dense: UniversalConfig = serde_saphyr::from_str(yaml).unwrap();
        assert!(mla_refusal(&dense).is_none());
        let mla: UniversalConfig = serde_saphyr::from_str(&format!(
            "{yaml}  kv_latent_dim: 8\n  q_latent_dim: 6\n  d_rope: 4\n"
        ))
        .unwrap();
        let reason = mla_refusal(&mla).expect("refused");
        assert!(reason.contains("kv_latent_dim"), "{reason}");
    }
}

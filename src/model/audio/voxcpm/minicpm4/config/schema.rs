//! [`MiniCpm4Config`]: the resolved fields, `base_lm`'s defaults, and the
//! `residual_lm` override rule.

/// `config.json` sub-object holding `base_lm`'s architecture.
pub const DEFAULT_CONFIG_SECTION: &str = "lm_config";

/// Top-level `config.json` key holding `residual_lm`'s layer count.
pub const RESIDUAL_LM_NUM_LAYERS_KEY: &str = "residual_lm_num_layers";

/// Top-level `config.json` key holding `residual_lm`'s NoPE switch.
pub const RESIDUAL_LM_NO_ROPE_KEY: &str = "residual_lm_no_rope";

/// Resolved config for
/// [`MiniCpm4Model`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Model).
///
/// Read from a single `config.json` sub-object (`lm_config` for `base_lm`)
/// by [`MiniCpm4Config::from_config_json_section`], or derived from
/// `lm_config` plus the top-level `residual_lm_*` keys by
/// [`MiniCpm4Config::residual_lm_from_config_json`].
#[derive(Debug, Clone)]
pub struct MiniCpm4Config {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// Per-head width, read from the checkpoint's `kv_channels` — NEVER
    /// derived as `hidden_size / num_heads`. The two agree on `base_lm`
    /// (2048/16 == 128 == `kv_channels`), which is exactly what makes the
    /// derivation a latent bug: the local-encoder sibling has `hidden_dim`
    /// 1024 with `kv_channels` 128, where deriving gives 64 and silently
    /// mis-shapes every projection.
    pub head_dim: usize,
    /// Rows in the `embed_tokens` table. `0` means the checkpoint carries no
    /// embedding table at all (VoxCPM2's `residual_lm`), and the loader then
    /// builds a model whose
    /// [`embed`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::embed)
    /// errors instead of silently returning zeros.
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    /// Per-dimension LongRoPE short-context rescale, length `head_dim / 2`.
    pub rope_short_factor: Vec<f32>,
    /// Per-dimension LongRoPE long-context rescale, length `head_dim / 2`.
    /// `RoPE::precompute_freqs` selects this over `rope_short_factor` only
    /// when `max_position_embeddings > original_max_position_embeddings`;
    /// on this checkpoint the two are equal (32768 == 32768), so
    /// `rope_short_factor` is always selected in practice and the LongRoPE
    /// `attention_scaling` collapses to 1.0.
    pub rope_long_factor: Vec<f32>,
    /// NoPE: run this instantiation with NO rotary embedding at all
    /// (`residual_lm`). `true` makes the loader skip building a RoPE cache and
    /// makes every attention block skip the rotation on BOTH the full-sequence
    /// and the KV-cached path.
    ///
    /// Nothing takes RoPE's place — no ALiBi, no learned positions. Position
    /// then reaches the block only through the causal mask.
    pub no_rope: bool,
}

impl Default for MiniCpm4Config {
    /// `base_lm`'s architecture constants, verified against the VoxCPM2
    /// checkpoint (254 tensors: 1 embedding + 28 x 9 per-layer + 1 final
    /// norm).
    ///
    /// `rope_short_factor`/`rope_long_factor` are the RoPE IDENTITY
    /// (all-ones) here, NOT the checkpoint's real per-dimension values —
    /// those must come from the checkpoint's `config.json` via
    /// [`MiniCpm4Config::from_config_json`]. Using this `Default` as-is
    /// silently applies unscaled RoPE, which is numerically wrong for this
    /// checkpoint.
    fn default() -> Self {
        let head_dim = 128;
        Self {
            num_layers: 28,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim,
            vocab_size: 73448,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 32768,
            original_max_position_embeddings: 32768,
            rope_short_factor: vec![1.0; head_dim / 2],
            rope_long_factor: vec![1.0; head_dim / 2],
            no_rope: false,
        }
    }
}

impl MiniCpm4Config {
    /// Apply `residual_lm`'s three overrides to a parsed `lm_config`.
    ///
    /// Split out from [`residual_lm_from_config_json`](Self::residual_lm_from_config_json)
    /// so the override rule is testable without a file, and so a caller that
    /// already holds the `base_lm` config does not re-read the JSON.
    ///
    /// `vocab_size` drops to `0` — `residual_lm` is fed pre-computed
    /// embeddings and the checkpoint carries neither `embed_tokens` nor
    /// `lm_head` for it.
    pub fn into_residual_lm(mut self, num_layers: usize, no_rope: bool) -> Self {
        self.num_layers = num_layers;
        self.vocab_size = 0;
        self.no_rope = no_rope;
        self
    }

    /// Whether this instantiation owns an `embed_tokens` table.
    ///
    /// `false` for a `vocab_size == 0` config (`residual_lm`), which is fed
    /// pre-computed embeddings only.
    pub fn has_embedding(&self) -> bool {
        self.vocab_size > 0
    }

    /// Whether this instantiation rotates Q/K.
    ///
    /// `false` for a NoPE config (`residual_lm`), for which the loader builds
    /// no RoPE cache at all.
    pub fn uses_rope(&self) -> bool {
        !self.no_rope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_head_dim_matches_short_factor_len() {
        let cfg = MiniCpm4Config::default();
        assert_eq!(cfg.rope_short_factor.len(), cfg.head_dim / 2);
        assert_eq!(cfg.rope_long_factor.len(), cfg.head_dim / 2);
        assert!(cfg.has_embedding());
    }
}

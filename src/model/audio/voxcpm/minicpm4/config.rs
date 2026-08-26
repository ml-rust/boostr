//! Configuration for VoxCPM2's MiniCPM4 decoder-only transformer, resolved
//! from the checkpoint's `config.json`.
//!
//! Every architectural knob is config-driven on purpose: VoxCPM2's
//! `residual_lm` is the SAME architecture as `base_lm` with a different
//! config (8 layers, `vocab_size` 0 hence no `embed_tokens` table, and no
//! RoPE), so it becomes a second [`MiniCpm4Config`] rather than a forked
//! module.
//!
//! `residual_lm` has NO section of its own in `config.json`. The reference
//! deep-copies `lm_config` and overrides three fields, so
//! [`MiniCpm4Config::residual_lm_from_config_json`] does the same here,
//! reading the two top-level `residual_lm_*` keys.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::Path;

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
    /// Parse `lm_config` out of a VoxCPM2 `config.json` — i.e. `base_lm`.
    pub fn from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_config_json_section(path, DEFAULT_CONFIG_SECTION)
    }

    /// Parse an arbitrary sub-object of a VoxCPM2 `config.json`.
    ///
    /// `section` is a top-level key (`"lm_config"` for `base_lm`). It is a
    /// parameter so a checkpoint that DOES carry a second architecture
    /// sub-object needs no second parser.
    ///
    /// This does NOT reach `residual_lm`: that checkpoint has no
    /// `residual_lm_config` section to name. Use
    /// [`residual_lm_from_config_json`](Self::residual_lm_from_config_json).
    pub fn from_config_json_section<P: AsRef<Path>>(path: P, section: &str) -> Result<Self> {
        Self::from_root(&read_config_root(path)?, section)
    }

    /// Resolve `residual_lm`'s config from a VoxCPM2 `config.json`.
    ///
    /// `residual_lm` has no sub-object of its own. The reference deep-copies
    /// `lm_config` and overrides exactly three fields
    /// (`voxcpm2.py:189-193`), and this reproduces that:
    ///
    /// | field | value |
    /// | --- | --- |
    /// | `num_layers` | top-level `residual_lm_num_layers` |
    /// | `no_rope` | top-level `residual_lm_no_rope` |
    /// | `vocab_size` | `0` — no `embed_tokens`, no `lm_head` |
    ///
    /// Both top-level keys are REQUIRED. Defaulting a missing
    /// `residual_lm_no_rope` to `false` would silently rotate a stack the
    /// reference never rotates, staying shape-valid while computing a
    /// different model.
    pub fn residual_lm_from_config_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = read_config_root(path)?;
        let base = Self::from_root(&root, DEFAULT_CONFIG_SECTION)?;
        let num_layers = root
            .get(RESIDUAL_LM_NUM_LAYERS_KEY)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "VoxCPM2 config.json has no integer `{RESIDUAL_LM_NUM_LAYERS_KEY}`; \
                     residual_lm has no config section of its own, so its layer count \
                     can only come from that top-level key"
                ),
            })? as usize;
        let no_rope = root
            .get(RESIDUAL_LM_NO_ROPE_KEY)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "VoxCPM2 config.json has no boolean `{RESIDUAL_LM_NO_ROPE_KEY}`; \
                     guessing it would silently apply or drop RoPE"
                ),
            })?;
        Ok(base.into_residual_lm(num_layers, no_rope))
    }

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

    /// Resolve one architecture sub-object out of an already-parsed root.
    fn from_root(root: &serde_json::Value, section: &str) -> Result<Self> {
        let sub = root.get(section).ok_or_else(|| Error::ModelError {
            reason: format!("VoxCPM2 config.json has no `{section}` object"),
        })?;
        let raw: RawLmConfig =
            serde_json::from_value(sub.clone()).map_err(|e| Error::ModelError {
                reason: format!("invalid VoxCPM2 config.json `{section}`: {e}"),
            })?;
        raw.resolve()
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

/// Read and parse a VoxCPM2 `config.json` once.
fn read_config_root<P: AsRef<Path>>(path: P) -> Result<serde_json::Value> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::ModelError {
        reason: format!("failed to read {}: {e}", path.as_ref().display()),
    })?;
    serde_json::from_str(&content).map_err(|e| Error::ModelError {
        reason: format!("invalid VoxCPM2 config.json: {e}"),
    })
}

/// One MiniCPM4 config sub-object.
///
/// Field names are the HuggingFace spellings, read verbatim from the
/// checkpoint's `lm_config`: `num_hidden_layers`, `hidden_size`,
/// `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
/// `kv_channels`, `vocab_size`, `rms_norm_eps`, `rope_theta`,
/// `max_position_embeddings`, `use_mup`. `original_max_position_embeddings`
/// lives inside the nested `rope_scaling` object, not alongside them.
#[derive(Debug, Deserialize)]
struct RawLmConfig {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    /// Per-head width. Distinct from `hidden_size / num_attention_heads` —
    /// see [`MiniCpm4Config::head_dim`].
    kv_channels: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: usize,
    rope_scaling: RawRopeScaling,
    /// muP parameterization. `false` on this checkpoint. When it is `true`
    /// the decoder residual carries a `scale_depth / sqrt(num_layers)`
    /// factor and the embedding output is multiplied by `scale_emb`, none of
    /// which this port implements — so a `true` here is rejected rather than
    /// silently ignored.
    #[serde(default)]
    use_mup: bool,
}

#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
}

impl RawLmConfig {
    fn resolve(self) -> Result<MiniCpm4Config> {
        if self.use_mup {
            return Err(Error::ModelError {
                reason: "use_mup=true is not supported: the muP residual scaling \
                         (scale_depth/sqrt(num_layers)) and scale_emb are not \
                         implemented, and ignoring them computes a different model"
                    .to_string(),
            });
        }
        let half_dim = self.kv_channels / 2;
        if self.rope_scaling.short_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.short_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.rope_scaling.short_factor.len()
                ),
            });
        }
        if self.rope_scaling.long_factor.len() != half_dim {
            return Err(Error::ModelError {
                reason: format!(
                    "rope_scaling.long_factor has {} entries, expected {half_dim} \
                     (kv_channels/2)",
                    self.rope_scaling.long_factor.len()
                ),
            });
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::ModelError {
                reason: format!(
                    "num_attention_heads ({}) must be a nonzero multiple of \
                     num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }
        Ok(MiniCpm4Config {
            num_layers: self.num_hidden_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.kv_channels,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            original_max_position_embeddings: self
                .rope_scaling
                .original_max_position_embeddings
                .unwrap_or(self.max_position_embeddings),
            rope_short_factor: self.rope_scaling.short_factor,
            rope_long_factor: self.rope_scaling.long_factor,
            // `lm_config` carries no `no_rope`: NoPE is a `residual_lm`
            // override applied by `into_residual_lm`, never a parsed field.
            no_rope: false,
        })
    }
}

#[cfg(test)]
mod tests;

//! Hyperparameters of the Qwen3-VL vision tower, read from `clip.*` keys.

use crate::error::{Error, Result};
use crate::format::gguf::GgufMetadata;

/// Projector type string this encoder accepts.
pub const PROJECTOR_TYPE: &str = "qwen3vl_merger";

/// Vision tower and merger sizes.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlVisionConfig {
    /// Side of the square training grid in pixels. The learned position table
    /// covers `(image_size / patch_size)^2` cells.
    pub image_size: usize,
    /// Side of one patch in pixels.
    pub patch_size: usize,
    /// Width of one patch token.
    pub hidden_size: usize,
    /// Width of the feed-forward hidden layer.
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Side of the square of patches merged into one output token.
    pub spatial_merge: usize,
    /// Width of one output token after the merger.
    pub projection_dim: usize,
    /// Epsilon of every layer norm.
    pub layer_norm_eps: f32,
    /// Per-channel mean subtracted after scaling pixels to `[0, 1]`.
    pub image_mean: [f32; 3],
    /// Per-channel divisor applied after the mean.
    pub image_std: [f32; 3],
    /// Smallest number of output tokens an image resizes up to.
    pub image_min_tokens: usize,
    /// Largest number of output tokens an image resizes down to.
    pub image_max_tokens: usize,
}

impl Qwen3VlVisionConfig {
    /// Read the tower sizes from an mmproj header.
    ///
    /// Fails when `general.architecture` is not `clip`, when
    /// `clip.projector_type` is not [`PROJECTOR_TYPE`], or when any size key
    /// is missing. Token limits default to 8 and 4096.
    pub fn from_gguf(meta: &GgufMetadata) -> Result<Self> {
        let arch = meta.architecture().unwrap_or("");
        if arch != "clip" {
            return Err(Error::ModelError {
                reason: format!("qwen3vl vision: general.architecture is '{arch}', want 'clip'"),
            });
        }
        let proj = meta.get_string("clip.projector_type").unwrap_or("");
        if proj != PROJECTOR_TYPE {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen3vl vision: clip.projector_type is '{proj}', want '{PROJECTOR_TYPE}'"
                ),
            });
        }
        let u32_key = |key: &str| -> Result<usize> {
            meta.get_u32(key)
                .map(|v| v as usize)
                .ok_or_else(|| Error::ModelError {
                    reason: format!("qwen3vl vision: metadata key '{key}' missing"),
                })
        };
        let image_size = u32_key("clip.vision.image_size")?;
        let patch_size = u32_key("clip.vision.patch_size")?;
        let hidden_size = u32_key("clip.vision.embedding_length")?;
        let intermediate_size = u32_key("clip.vision.feed_forward_length")?;
        let num_layers = u32_key("clip.vision.block_count")?;
        let num_heads = u32_key("clip.vision.attention.head_count")?;
        let projection_dim = u32_key("clip.vision.projection_dim")?;
        let spatial_merge = meta
            .get_u32("clip.vision.spatial_merge_size")
            .map(|v| v as usize)
            .unwrap_or(2);
        let layer_norm_eps = meta
            .get_f32("clip.vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-6);
        let image_mean = f32_triple(meta, "clip.vision.image_mean")?;
        let image_std = f32_triple(meta, "clip.vision.image_std")?;

        let cfg = Self {
            image_size,
            patch_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            spatial_merge,
            projection_dim,
            layer_norm_eps,
            image_mean,
            image_std,
            image_min_tokens: 8,
            image_max_tokens: 4096,
        };
        cfg.check()?;
        Ok(cfg)
    }

    /// Reject sizes the graph cannot run.
    pub fn check(&self) -> Result<()> {
        let bad = |what: &str| Error::ModelError {
            reason: format!("qwen3vl vision: {what}"),
        };
        if self.patch_size == 0 || !self.image_size.is_multiple_of(self.patch_size) {
            return Err(bad(&format!(
                "image_size {} is not a multiple of patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(bad(&format!(
                "hidden_size {} is not a multiple of head_count {}",
                self.hidden_size, self.num_heads
            )));
        }
        if !self.head_dim().is_multiple_of(4) {
            return Err(bad(&format!(
                "head_dim {} is not a multiple of 4 (2D rope needs four equal sections)",
                self.head_dim()
            )));
        }
        if self.spatial_merge != 2 {
            return Err(bad(&format!(
                "spatial_merge_size {} unsupported, want 2",
                self.spatial_merge
            )));
        }
        if self.image_min_tokens == 0 || self.image_max_tokens < self.image_min_tokens {
            return Err(bad(&format!(
                "token limits min {} max {} invalid",
                self.image_min_tokens, self.image_max_tokens
            )));
        }
        Ok(())
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Pixels covered by one output token on a side.
    pub fn align(&self) -> usize {
        self.patch_size * self.spatial_merge
    }

    /// Pixel area of one output token.
    pub fn token_area(&self) -> usize {
        self.align() * self.align()
    }

    /// Smallest pixel area after resize.
    pub fn min_pixels(&self) -> usize {
        self.image_min_tokens * self.token_area()
    }

    /// Largest pixel area after resize.
    pub fn max_pixels(&self) -> usize {
        self.image_max_tokens * self.token_area()
    }

    /// Side of the learned position grid in patches.
    pub fn pos_grid_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Width of one merger input row: `spatial_merge^2` patch tokens.
    pub fn merged_width(&self) -> usize {
        self.hidden_size * self.spatial_merge * self.spatial_merge
    }
}

fn f32_triple(meta: &GgufMetadata, key: &str) -> Result<[f32; 3]> {
    let arr = meta.get_array(key).ok_or_else(|| Error::ModelError {
        reason: format!("qwen3vl vision: metadata key '{key}' missing"),
    })?;
    if arr.len() != 3 {
        return Err(Error::ModelError {
            reason: format!(
                "qwen3vl vision: metadata key '{key}' has {} entries, want 3",
                arr.len()
            ),
        });
    }
    let mut out = [0f32; 3];
    for (dst, v) in out.iter_mut().zip(arr) {
        *dst = v.as_f32().ok_or_else(|| Error::ModelError {
            reason: format!("qwen3vl vision: metadata key '{key}' entry is not f32"),
        })?;
    }
    Ok(out)
}

/// Sizes of the shipped mmproj, for tests that need a config without a file.
#[cfg(test)]
pub(crate) fn bonsai2_test_config() -> Qwen3VlVisionConfig {
    Qwen3VlVisionConfig {
        image_size: 768,
        patch_size: 16,
        hidden_size: 1152,
        intermediate_size: 4304,
        num_layers: 27,
        num_heads: 16,
        spatial_merge: 2,
        projection_dim: 5120,
        layer_norm_eps: 1e-6,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        image_min_tokens: 8,
        image_max_tokens: 4096,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_sizes() {
        let cfg = bonsai2_test_config();
        assert!(cfg.check().is_ok());
        assert_eq!(cfg.head_dim(), 72);
        assert_eq!(cfg.align(), 32);
        assert_eq!(cfg.min_pixels(), 8 * 1024);
        assert_eq!(cfg.max_pixels(), 4096 * 1024);
        assert_eq!(cfg.pos_grid_side(), 48);
        assert_eq!(cfg.merged_width(), 4608);
    }

    #[test]
    fn check_rejects_bad_head_split() {
        let mut cfg = bonsai2_test_config();
        cfg.num_heads = 7;
        assert!(cfg.check().is_err());
    }
}

//! [`RrdbConfig`]: the shape of an RRDBNet and the state-dict keys it reads.

use crate::error::{Error, Result};

/// The only overall upscale factor [`super::RrdbNet`] runs.
pub const RRDB_SUPPORTED_SCALE: usize = 2;

/// Pixel-unshuffle factor of the head at [`RRDB_SUPPORTED_SCALE`]. The head
/// folds each 2×2 patch into channels, and the two ×2 upsamples restore
/// four times the feature resolution.
pub const RRDB_UNSHUFFLE_FACTOR: usize = 2;

/// Convs in one residual dense block.
pub const RDB_CONVS: usize = 5;

/// Residual dense blocks in one RRDB.
pub const RRDB_DENSE_BLOCKS: usize = 3;

/// Shape of an RRDBNet network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrdbConfig {
    /// Input image channels.
    pub num_in_ch: usize,
    /// Output image channels.
    pub num_out_ch: usize,
    /// Feature channels of the trunk.
    pub num_feat: usize,
    /// RRDB blocks in the body.
    pub num_block: usize,
    /// Channels each of `conv1..conv4` of a dense block adds.
    pub num_grow_ch: usize,
    /// Overall upscale factor. Only [`RRDB_SUPPORTED_SCALE`] loads.
    pub scale: usize,
}

impl RrdbConfig {
    /// The `RealESRGAN_x2plus` network: RGB in and out, 64 features,
    /// 23 RRDB blocks, growth 32, ×2. `2x_PureVision` has the same shape.
    pub fn realesrgan_x2plus() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 64,
            num_block: 23,
            num_grow_ch: 32,
            scale: RRDB_SUPPORTED_SCALE,
        }
    }

    /// Check every field is positive and `scale` is supported.
    pub fn validate(&self) -> Result<()> {
        let fields = [
            ("num_in_ch", self.num_in_ch),
            ("num_out_ch", self.num_out_ch),
            ("num_feat", self.num_feat),
            ("num_block", self.num_block),
            ("num_grow_ch", self.num_grow_ch),
        ];
        if let Some((field, _)) = fields.iter().find(|(_, v)| *v == 0) {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!("RRDBNet {field} is 0 in {self:?}; it must be at least 1"),
            });
        }
        if self.scale != RRDB_SUPPORTED_SCALE {
            return Err(Error::InvalidArgument {
                arg: "config",
                reason: format!(
                    "RRDBNet scale {} is not supported; pass scale {RRDB_SUPPORTED_SCALE}, the only one implemented",
                    self.scale
                ),
            });
        }
        Ok(())
    }

    /// Input channels of `conv_first`: the image channels after the
    /// pixel-unshuffle head.
    pub fn first_conv_in_channels(&self) -> usize {
        self.num_in_ch * RRDB_UNSHUFFLE_FACTOR * RRDB_UNSHUFFLE_FACTOR
    }

    /// Input channels of `conv{k}` of a dense block, `k` in `1..=5`:
    /// `num_feat + (k - 1) · num_grow_ch`. `k = 0` counts as `k = 1`.
    pub fn dense_conv_in_channels(&self, k: usize) -> usize {
        self.num_feat + k.saturating_sub(1) * self.num_grow_ch
    }

    /// Channels of one dense-block feature buffer: the block input plus
    /// the outputs of `conv1..conv4`. `conv5` reads all of them.
    pub fn dense_channels(&self) -> usize {
        self.dense_conv_in_channels(RDB_CONVS)
    }

    /// Every state-dict key the network reads, in forward order.
    ///
    /// `conv_first`, then `body.{b}.rdb{d}.conv{c}` for every block `b`,
    /// dense block `d` in `1..=3` and conv `c` in `1..=5`, then
    /// `conv_body`, `conv_up1`, `conv_up2`, `conv_hr` and `conv_last`. Each
    /// conv has `.weight` and `.bias`.
    pub fn expected_keys(&self) -> Vec<String> {
        let tail = ["conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"];
        let dense: Vec<String> = (0..self.num_block)
            .flat_map(|b| {
                (1..=RRDB_DENSE_BLOCKS).flat_map(move |d| {
                    (1..=RDB_CONVS).map(move |c| format!("body.{b}.rdb{d}.conv{c}"))
                })
            })
            .collect();
        std::iter::once("conv_first".to_string())
            .chain(dense)
            .chain(tail.iter().map(|s| s.to_string()))
            .flat_map(|conv| [format!("{conv}.weight"), format!("{conv}.bias")])
            .collect()
    }
}

impl Default for RrdbConfig {
    fn default() -> Self {
        Self::realesrgan_x2plus()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x2plus_has_351_convs() {
        let keys = RrdbConfig::realesrgan_x2plus().expected_keys();
        assert_eq!(keys.len(), 2 * (6 + 23 * 15));
        assert_eq!(keys[0], "conv_first.weight");
        assert_eq!(keys[2], "body.0.rdb1.conv1.weight");
        assert!(keys.iter().any(|k| k == "body.22.rdb3.conv5.bias"));
        assert!(!keys.iter().any(|k| k.starts_with("body.23.")));
        assert_eq!(keys[keys.len() - 1], "conv_last.bias");
    }

    #[test]
    fn dense_channels_follow_growth() {
        let config = RrdbConfig::realesrgan_x2plus();
        assert_eq!(config.first_conv_in_channels(), 12);
        assert_eq!(config.dense_conv_in_channels(1), 64);
        assert_eq!(config.dense_conv_in_channels(4), 160);
        assert_eq!(config.dense_channels(), 192);
    }

    #[test]
    fn only_scale_2_is_accepted() {
        let config = RrdbConfig {
            scale: 4,
            ..RrdbConfig::default()
        };
        let err = config.validate().err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|e| e.contains("scale 4") && e.contains("pass scale 2")),
            "{err:?}"
        );
    }

    #[test]
    fn zero_field_is_rejected() {
        let config = RrdbConfig {
            num_grow_ch: 0,
            ..RrdbConfig::default()
        };
        let err = config.validate().err().map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|e| e.contains("num_grow_ch")),
            "{err:?}"
        );
    }
}

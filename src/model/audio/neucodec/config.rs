//! Configuration for the NeuCodec acoustic decoder.
//!
//! All defaults are the VERIFIED values dumped from the real
//! `neuphonic/neucodec` `model.safetensors` header (see the module's
//! architecture doc comment in `decoder.rs`) — NOT the (partly wrong)
//! `config.json` or GitHub source.

use crate::error::{Error, Result};

/// Dimensions and hyperparameters for [`super::decoder::NeuCodecDecoder`].
///
/// There are deliberately no RoPE fields, despite `config.json` advertising
/// `rope_parameters`: the released decoder's rotary embedding is a verified
/// no-op (see `transformer_block`'s module doc), so this port applies no
/// positional encoding and has nothing to parameterize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuCodecDecoderConfig {
    /// Residual-stream width used by `embed`, the resnet blocks, and every
    /// transformer block (checkpoint: 1024).
    pub hidden_size: usize,
    /// Input feature width fed to `fc` — the FSQ `project_out` dimension
    /// (checkpoint: 2048).
    pub fc_in_dim: usize,
    /// Kernel size of the `embed` Conv1d (checkpoint: 7).
    pub embed_kernel_size: usize,
    /// Kernel size of `conv1`/`conv2` inside each `ResnetBlock` (checkpoint: 3).
    pub resnet_kernel_size: usize,
    /// Number of `ResnetBlock`s in `prior_net` (checkpoint: 2).
    pub num_prior_resnet_blocks: usize,
    /// Number of `ResnetBlock`s in `post_net` (checkpoint: 2).
    pub num_post_resnet_blocks: usize,
    /// Number of `TransformerBlock`s (checkpoint: 12).
    pub num_transformer_layers: usize,
    /// Attention heads per transformer block (checkpoint: 16).
    pub num_heads: usize,
    /// Per-head dimension (checkpoint: 64, so `num_heads * head_dim == hidden_size`).
    pub head_dim: usize,
    /// Hidden width of the plain 2-layer transformer MLP (checkpoint: 4096).
    pub mlp_intermediate_size: usize,
    /// Epsilon for the RMSNorm layers inside each `TransformerBlock`.
    pub rms_norm_eps: f32,
    /// Number of groups in each `ResnetBlock`'s GroupNorm layers. Upstream
    /// `Normalize(in_channels, num_groups=32)`; must divide `hidden_size`.
    pub resnet_norm_groups: usize,
    /// Epsilon for the `ResnetBlock` GroupNorm layers (upstream `eps=1e-6`,
    /// NOT PyTorch's `1e-5` GroupNorm default).
    pub resnet_norm_eps: f32,
    /// Epsilon for the post-transformer `norm` LayerNorm. Upstream:
    /// `self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)`.
    pub layer_norm_eps: f32,
    /// ISTFT FFT size — upstream `ISTFTHead(n_fft=hop_length * 4)`, and
    /// `out_dim = n_fft + 2`. The verified `head.linear` output width of 1922
    /// therefore pins `n_fft = 1920`.
    pub n_fft: usize,
    /// ISTFT hop length. `n_fft = hop_length * 4` upstream, so `n_fft = 1920`
    /// gives `hop_length = 480` — NOT the 320 default in the upstream
    /// `CodecDecoderVocos.__init__` signature, which does not match this
    /// checkpoint (320 would imply `n_fft = 1280` and a 1282-wide head).
    ///
    /// 480 is also the value consistent with NeuCodec's documented rates: at
    /// the 24 kHz output rate the decoder's latent frame rate is
    /// `24000 / 480 = 50` Hz.
    pub hop_length: usize,
    /// Upper clamp applied to the head's LINEAR magnitude, i.e. AFTER `exp()`
    /// (upstream: `mag = torch.exp(mag); mag = torch.clip(mag, max=1e2)`).
    pub mag_clamp_max: f32,
}

impl Default for NeuCodecDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            fc_in_dim: 2048,
            embed_kernel_size: 7,
            resnet_kernel_size: 3,
            num_prior_resnet_blocks: 2,
            num_post_resnet_blocks: 2,
            num_transformer_layers: 12,
            num_heads: 16,
            head_dim: 64,
            mlp_intermediate_size: 4096,
            rms_norm_eps: 1e-6,
            resnet_norm_groups: 32,
            resnet_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
            n_fft: 1920,
            hop_length: 480,
            mag_clamp_max: 1e2,
        }
    }
}

impl NeuCodecDecoderConfig {
    /// `n_fft / 2 + 1` — number of magnitude/phase frequency bins.
    pub fn n_freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// `head.linear` output width: `2 * n_freq_bins` (checkpoint: 1922).
    pub fn head_out_dim(&self) -> usize {
        2 * self.n_freq_bins()
    }

    /// Validate internal consistency of the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "hidden_size",
                reason: "must be > 0".into(),
            });
        }
        if self.fc_in_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "fc_in_dim",
                reason: "must be > 0".into(),
            });
        }
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_heads/head_dim",
                reason: "must both be > 0".into(),
            });
        }
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "num_heads ({}) * head_dim ({}) = {} must equal hidden_size ({})",
                    self.num_heads,
                    self.head_dim,
                    self.num_heads * self.head_dim,
                    self.hidden_size
                ),
            });
        }
        if self.mlp_intermediate_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "mlp_intermediate_size",
                reason: "must be > 0".into(),
            });
        }
        if self.embed_kernel_size == 0 || self.embed_kernel_size.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "embed_kernel_size",
                reason: "must be odd and > 0 (same-padding convolution)".into(),
            });
        }
        if self.resnet_kernel_size == 0 || self.resnet_kernel_size.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "resnet_kernel_size",
                reason: "must be odd and > 0 (same-padding convolution)".into(),
            });
        }
        if self.resnet_norm_groups == 0 {
            return Err(Error::InvalidArgument {
                arg: "resnet_norm_groups",
                reason: "must be > 0".into(),
            });
        }
        if !self.hidden_size.is_multiple_of(self.resnet_norm_groups) {
            return Err(Error::InvalidArgument {
                arg: "resnet_norm_groups",
                reason: format!(
                    "must divide hidden_size ({}), got {}",
                    self.hidden_size, self.resnet_norm_groups
                ),
            });
        }
        if self.num_transformer_layers == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_transformer_layers",
                reason: "must be > 0".into(),
            });
        }
        if self.n_fft == 0 || !self.n_fft.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "n_fft",
                reason: "must be a positive even number".into(),
            });
        }
        if self.hop_length == 0 {
            return Err(Error::InvalidArgument {
                arg: "hop_length",
                reason: "must be > 0".into(),
            });
        }
        if self.hop_length > self.n_fft {
            return Err(Error::InvalidArgument {
                arg: "hop_length",
                reason: format!(
                    "must not exceed n_fft ({}), got {}",
                    self.n_fft, self.hop_length
                ),
            });
        }
        // The Vocos `padding="same"` ISTFT trims `(n_fft - hop)/2` per end. If
        // that difference is odd the floor makes the trim asymmetric, and the
        // decoder emits `frames*hop + 1` samples instead of exactly one hop per
        // frame. The released config satisfies this (1920 - 480 = 1440).
        if !(self.n_fft - self.hop_length).is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "hop_length",
                reason: format!(
                    "n_fft - hop_length must be even for one hop of audio per frame, \
                     got {} - {} = {}",
                    self.n_fft,
                    self.hop_length,
                    self.n_fft - self.hop_length
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_verified_checkpoint_shapes() {
        let cfg = NeuCodecDecoderConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.fc_in_dim, 2048);
        assert_eq!(cfg.n_freq_bins(), 961);
        assert_eq!(cfg.head_out_dim(), 1922);
        assert!(cfg.validate().is_ok());
    }

    /// Upstream ties the head geometry together: `n_fft = hop_length * 4` and
    /// `out_dim = n_fft + 2`. The checkpoint's 1922-wide `head.linear` is what
    /// pins `hop_length = 480` (the 320 in upstream's `__init__` default does
    /// not describe this checkpoint).
    #[test]
    fn hop_length_is_pinned_by_head_width() {
        let cfg = NeuCodecDecoderConfig::default();
        assert_eq!(cfg.n_fft, cfg.hop_length * 4, "upstream n_fft = hop * 4");
        assert_eq!(
            cfg.head_out_dim(),
            cfg.n_fft + 2,
            "upstream out_dim = n_fft + 2"
        );
        assert_eq!(cfg.hop_length, 480);
        // 24 kHz output / 480 = the documented 50 Hz NeuCodec frame rate.
        assert_eq!(24_000 / cfg.hop_length, 50);
    }

    /// An odd `n_fft - hop_length` makes the Vocos trim asymmetric, so the
    /// decoder would emit `frames*hop + 1` samples instead of one hop per frame.
    #[test]
    fn rejects_odd_gap_between_n_fft_and_hop() {
        let cfg = NeuCodecDecoderConfig {
            hop_length: 481,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_hop_longer_than_n_fft() {
        let cfg = NeuCodecDecoderConfig {
            hop_length: 4096,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_group_count_that_does_not_divide_hidden_size() {
        let cfg = NeuCodecDecoderConfig {
            resnet_norm_groups: 33,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_head_dim_mismatch() {
        let mut cfg = NeuCodecDecoderConfig {
            head_dim: 63,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        cfg.head_dim = 64;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_even_kernel_sizes() {
        let cfg = NeuCodecDecoderConfig {
            resnet_kernel_size: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_odd_n_fft() {
        let cfg = NeuCodecDecoderConfig {
            n_fft: 1921,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }
}

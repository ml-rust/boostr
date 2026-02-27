//! Mamba2 layer configuration.

use crate::error::{Error, Result};

/// Configuration for a Mamba2 layer.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    /// Hidden dimension (model dimension)
    pub d_model: usize,
    /// State dimension (SSM state size, typically 64-128)
    pub d_state: usize,
    /// Number of SSM heads
    pub nheads: usize,
    /// Head dimension = d_model * expand / nheads
    pub headdim: usize,
    /// Expansion factor (typically 2)
    pub expand: usize,
    /// Number of groups for grouped B/C projections (must be 1 or nheads)
    pub ngroups: usize,
    /// Convolution kernel size (typically 4)
    pub d_conv: usize,
    /// Chunk size for chunked SSD (typically 64 or 128)
    pub chunk_size: usize,
    /// Whether to use dt bias
    pub use_dt_bias: bool,
    /// Whether to use D skip connection
    pub use_d: bool,
    /// Apply softplus to dt
    pub dt_softplus: bool,
}

impl Mamba2Config {
    /// Create default config for a given hidden dimension.
    pub fn new(d_model: usize) -> Self {
        let expand = 2;
        let nheads = (d_model / 64).max(1);
        let headdim = d_model * expand / nheads;
        Self {
            d_model,
            d_state: 64,
            nheads,
            headdim,
            expand,
            ngroups: 1,
            d_conv: 4,
            chunk_size: 64,
            use_dt_bias: true,
            use_d: true,
            dt_softplus: true,
        }
    }

    /// Set the number of SSM heads (recomputes headdim).
    pub fn with_nheads(mut self, nheads: usize) -> Self {
        self.nheads = nheads;
        self.headdim = self.d_model * self.expand / nheads;
        self
    }

    /// Set the SSM state dimension per head.
    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    /// Set the expansion factor for inner dimension (recomputes headdim).
    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self.headdim = self.d_model * expand / self.nheads;
        self
    }

    /// Set the depthwise convolution kernel size.
    pub fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    /// Set the chunk size for chunked SSM scan.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable or disable softplus on dt (timestep) values.
    pub fn with_dt_softplus(mut self, dt_softplus: bool) -> Self {
        self.dt_softplus = dt_softplus;
        self
    }

    /// Enable or disable the D (skip connection) parameter.
    pub fn with_use_d(mut self, use_d: bool) -> Self {
        self.use_d = use_d;
        self
    }

    /// Enable or disable the dt bias parameter.
    pub fn with_use_dt_bias(mut self, use_dt_bias: bool) -> Self {
        self.use_dt_bias = use_dt_bias;
        self
    }

    /// Create a `Mamba2Config` from a `UniversalConfig`'s SSM section.
    pub fn from_universal(
        config: &crate::model::config::UniversalConfig,
    ) -> crate::error::Result<Self> {
        let ssm = config
            .ssm
            .as_ref()
            .ok_or_else(|| crate::error::Error::ModelError {
                reason: "Mamba2 requires ssm config section".into(),
            })?;
        let expand = ssm.expand;
        let nheads = ssm.num_heads;
        let headdim = ssm.head_dim;
        Ok(Self {
            d_model: config.hidden_size,
            d_state: ssm.state_size,
            nheads,
            headdim,
            expand,
            ngroups: ssm.n_groups,
            d_conv: ssm.conv_kernel,
            chunk_size: ssm.chunk_size,
            use_dt_bias: true,
            use_d: true,
            dt_softplus: true,
        })
    }

    /// Inner (expanded) dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Total projected dimension for in_proj output.
    pub fn proj_dim(&self) -> usize {
        2 * self.d_inner() + 2 * self.ngroups * self.d_state + self.nheads
    }

    /// Conv1d channels = d_inner + 2 * ngroups * d_state.
    pub fn conv_channels(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    /// Validate config constraints. Returns error if invalid.
    pub fn validate(&self) -> Result<()> {
        if self.ngroups != 1 && self.ngroups != self.nheads {
            return Err(Error::ModelError {
                reason: format!(
                    "ngroups must be 1 or nheads ({}), got {}",
                    self.nheads, self.ngroups
                ),
            });
        }
        if self.nheads == 0 || self.headdim == 0 {
            return Err(Error::ModelError {
                reason: "nheads and headdim must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba2_config_defaults() {
        let config = Mamba2Config::new(256);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.d_state, 64);
        assert_eq!(config.expand, 2);
        assert_eq!(config.d_inner(), 512);
        assert_eq!(config.nheads, 4);
        assert_eq!(config.headdim, 128);
        assert!(config.dt_softplus);
        assert!(config.use_dt_bias);
        assert!(config.use_d);
    }

    #[test]
    fn test_mamba2_config_builders() {
        let config = Mamba2Config::new(256)
            .with_nheads(8)
            .with_d_state(128)
            .with_expand(3);
        assert_eq!(config.nheads, 8);
        assert_eq!(config.d_state, 128);
        assert_eq!(config.expand, 3);
        assert_eq!(config.d_inner(), 768);
        assert_eq!(config.headdim, 96);
    }

    #[test]
    fn test_mamba2_config_validation() {
        let valid = Mamba2Config::new(256);
        assert!(valid.validate().is_ok());

        // ngroups=1 is valid
        let mut cfg = Mamba2Config::new(256);
        cfg.ngroups = 1;
        assert!(cfg.validate().is_ok());

        // ngroups=nheads is valid
        cfg.ngroups = cfg.nheads;
        assert!(cfg.validate().is_ok());

        // ngroups=2 with nheads=4 is invalid
        cfg.ngroups = 2;
        assert!(cfg.validate().is_err());
    }
}

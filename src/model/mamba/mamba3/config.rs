//! Mamba3 layer configuration.

use crate::error::{Error, Result};

/// Configuration for a Mamba3 layer.
#[derive(Debug, Clone)]
pub struct Mamba3Config {
    /// Hidden dimension (model dimension)
    pub d_model: usize,
    /// State dimension (SSM state size, must be even for complex RoPE)
    pub d_state: usize,
    /// Number of SSM heads
    pub nheads: usize,
    /// Head dimension = d_model * expand / nheads
    pub headdim: usize,
    /// Expansion factor (typically 2)
    pub expand: usize,
    /// Number of groups for grouped B/C projections (must be 1 or nheads)
    pub ngroups: usize,
    /// Convolution kernel size when `use_conv` is enabled
    pub d_conv: usize,
    /// Chunk size retained for config compatibility with SSD kernels
    pub chunk_size: usize,
    /// Apply softplus to dt
    pub dt_softplus: bool,
    /// Enable learned D skip connection
    pub use_d: bool,
    /// Enable learned dt bias
    pub use_dt_bias: bool,
    /// Enable Mamba3 complex-valued RoPE over B/C state channels
    pub use_complex_rope: bool,
    /// MIMO rank (0 = disabled / SISO)
    pub mimo_rank: usize,
    /// Enable optional causal Conv1D over x/B/C before the scan
    pub use_conv: bool,
    /// Minimum dt after softplus/clamp
    pub time_step_min: f64,
    /// Maximum dt after softplus/clamp
    pub time_step_max: f64,
}

impl Mamba3Config {
    /// Create a default Mamba3 config for a given hidden dimension.
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
            dt_softplus: true,
            use_d: true,
            use_dt_bias: true,
            use_complex_rope: false,
            mimo_rank: 0,
            use_conv: false,
            time_step_min: 0.001,
            time_step_max: 4.0,
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

    /// Set the expansion factor (recomputes headdim).
    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self.headdim = self.d_model * expand / self.nheads;
        self
    }

    /// Set the number of B/C groups.
    pub fn with_ngroups(mut self, ngroups: usize) -> Self {
        self.ngroups = ngroups;
        self
    }

    /// Set the optional convolution kernel size.
    pub fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    /// Set the chunk size retained for SSD compatibility.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable or disable complex RoPE.
    pub fn with_complex_rope(mut self, use_complex_rope: bool) -> Self {
        self.use_complex_rope = use_complex_rope;
        self
    }

    /// Set MIMO rank (0 disables MIMO).
    pub fn with_mimo_rank(mut self, mimo_rank: usize) -> Self {
        self.mimo_rank = mimo_rank;
        self
    }

    /// Enable or disable optional causal convolution.
    pub fn with_use_conv(mut self, use_conv: bool) -> Self {
        self.use_conv = use_conv;
        self
    }

    /// Enable or disable softplus on dt values.
    pub fn with_dt_softplus(mut self, dt_softplus: bool) -> Self {
        self.dt_softplus = dt_softplus;
        self
    }

    /// Enable or disable the D skip connection.
    pub fn with_use_d(mut self, use_d: bool) -> Self {
        self.use_d = use_d;
        self
    }

    /// Enable or disable dt bias.
    pub fn with_use_dt_bias(mut self, use_dt_bias: bool) -> Self {
        self.use_dt_bias = use_dt_bias;
        self
    }

    /// Create a `Mamba3Config` from a `UniversalConfig`'s SSM section.
    pub fn from_universal(
        config: &crate::model::config::UniversalConfig,
    ) -> crate::error::Result<Self> {
        let ssm = config
            .ssm
            .as_ref()
            .ok_or_else(|| crate::error::Error::ModelError {
                reason: "Mamba3 requires ssm config section".into(),
            })?;
        if !ssm.is_mamba3() {
            return Err(crate::error::Error::ModelError {
                reason: format!(
                    "Mamba3 requires ssm.variant = 'mamba3', got '{}'",
                    ssm.variant
                ),
            });
        }
        Ok(Self {
            d_model: config.hidden_size,
            d_state: ssm.state_size,
            nheads: ssm.num_heads,
            headdim: ssm.head_dim,
            expand: ssm.expand,
            ngroups: ssm.n_groups,
            d_conv: ssm.conv_kernel,
            chunk_size: ssm.chunk_size,
            dt_softplus: true,
            use_d: true,
            use_dt_bias: true,
            use_complex_rope: ssm.complex_rope.unwrap_or(false),
            mimo_rank: ssm.mimo_rank.unwrap_or(0),
            use_conv: ssm.use_conv.unwrap_or(false),
            time_step_min: 0.001,
            time_step_max: 4.0,
        })
    }

    /// Inner (expanded) dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Size of one grouped B or C projection before head repetition.
    pub fn bc_size(&self) -> usize {
        self.ngroups * self.d_state
    }

    /// Total projected dimension for in_proj output.
    pub fn proj_dim(&self) -> usize {
        2 * self.d_inner() + 2 * self.bc_size() + self.nheads
    }

    /// Conv1d channels = d_inner + B + C.
    pub fn conv_channels(&self) -> usize {
        self.d_inner() + 2 * self.bc_size()
    }

    /// Dimension processed by the SSM per head after optional MIMO up-projection.
    pub fn ssm_headdim(&self) -> usize {
        if self.mimo_rank > 0 {
            self.headdim * self.mimo_rank
        } else {
            self.headdim
        }
    }

    /// Validate config constraints. Returns error if invalid.
    pub fn validate(&self) -> Result<()> {
        if self.nheads == 0 || self.headdim == 0 {
            return Err(Error::ModelError {
                reason: "nheads and headdim must be > 0".to_string(),
            });
        }
        if self.d_state == 0 {
            return Err(Error::ModelError {
                reason: "d_state must be > 0".to_string(),
            });
        }
        if self.chunk_size == 0 {
            return Err(Error::ModelError {
                reason: "chunk_size must be > 0".to_string(),
            });
        }
        if self.use_conv && self.d_conv == 0 {
            return Err(Error::ModelError {
                reason: "d_conv must be > 0 when use_conv is true".to_string(),
            });
        }
        if self.ngroups != 1 && self.ngroups != self.nheads {
            return Err(Error::ModelError {
                reason: format!(
                    "ngroups must be 1 or nheads ({}), got {}",
                    self.nheads, self.ngroups
                ),
            });
        }
        if self.d_inner() != self.nheads * self.headdim {
            return Err(Error::ModelError {
                reason: format!(
                    "Mamba3 constraint violated: d_model * expand ({}) != nheads * headdim ({})",
                    self.d_inner(),
                    self.nheads * self.headdim
                ),
            });
        }
        if self.use_complex_rope && !self.d_state.is_multiple_of(2) {
            return Err(Error::ModelError {
                reason: format!(
                    "Mamba3 with complex RoPE requires even d_state, got {}",
                    self.d_state
                ),
            });
        }
        if self.time_step_min <= 0.0 || self.time_step_min >= self.time_step_max {
            return Err(Error::ModelError {
                reason: format!(
                    "invalid dt clamp range [{}, {}]",
                    self.time_step_min, self.time_step_max
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
    fn test_mamba3_config_defaults() {
        let config = Mamba3Config::new(256);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.d_state, 64);
        assert_eq!(config.expand, 2);
        assert_eq!(config.d_inner(), 512);
        assert_eq!(config.nheads, 4);
        assert_eq!(config.headdim, 128);
        assert!(!config.use_complex_rope);
        assert_eq!(config.mimo_rank, 0);
        assert!(!config.use_conv);
        assert!(config.dt_softplus);
        assert!(config.use_dt_bias);
        assert!(config.use_d);
    }

    #[test]
    fn test_mamba3_config_builders() {
        let config = Mamba3Config::new(256)
            .with_nheads(8)
            .with_d_state(128)
            .with_expand(3)
            .with_complex_rope(true)
            .with_mimo_rank(2)
            .with_use_conv(true);
        assert_eq!(config.nheads, 8);
        assert_eq!(config.d_state, 128);
        assert_eq!(config.expand, 3);
        assert_eq!(config.d_inner(), 768);
        assert_eq!(config.headdim, 96);
        assert!(config.use_complex_rope);
        assert_eq!(config.mimo_rank, 2);
        assert!(config.use_conv);
        assert_eq!(config.ssm_headdim(), 192);
    }

    #[test]
    fn test_mamba3_config_validation() {
        let valid = Mamba3Config::new(256);
        assert!(valid.validate().is_ok());

        let invalid_groups = Mamba3Config::new(256).with_ngroups(2);
        assert!(invalid_groups.validate().is_err());

        let odd_rope = Mamba3Config::new(256)
            .with_d_state(63)
            .with_complex_rope(true);
        assert!(odd_rope.validate().is_err());
    }
}

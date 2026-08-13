//! Mamba1 layer configuration.

use crate::error::{Error, Result};
use crate::model::mamba::mamba2::Mamba2Config;

/// Default recurrent hidden-state clamp from the oxidizr Mamba1 reference.
pub const DEFAULT_HIDDEN_STATE_CLAMP: f64 = 30.0;

/// Configuration for a Mamba1 layer.
#[derive(Debug, Clone)]
pub struct Mamba1Config {
    /// Hidden dimension (model dimension).
    pub d_model: usize,
    /// State dimension (SSM state size; classic Mamba1 uses 16).
    pub d_state: usize,
    /// Expansion factor (classic Mamba1 uses 2).
    pub expand: usize,
    /// Depthwise causal convolution kernel size.
    pub d_conv: usize,
    /// Chunk size retained only for shared SSM config compatibility.
    pub chunk_size: usize,
    /// Apply softplus to dt after `dt_proj`.
    pub dt_softplus: bool,
    /// Enable learned D skip connection.
    pub use_d: bool,
    /// Clamp recurrent hidden state after each sequential update.
    pub hidden_state_clamp: Option<f64>,
}

impl Mamba1Config {
    /// Create a default Mamba1 config for a given hidden dimension.
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16,
            expand: 2,
            d_conv: 4,
            chunk_size: 64,
            dt_softplus: true,
            use_d: true,
            hidden_state_clamp: Some(DEFAULT_HIDDEN_STATE_CLAMP),
        }
    }

    /// Set the SSM state dimension.
    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    /// Set the expansion factor.
    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self
    }

    /// Set the depthwise convolution kernel size.
    pub fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    /// Set the chunk size retained for shared SSM config compatibility.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
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

    /// Set the hidden-state clamp applied inside the sequential scan.
    pub fn with_hidden_state_clamp(mut self, hidden_state_clamp: Option<f64>) -> Self {
        self.hidden_state_clamp = hidden_state_clamp;
        self
    }

    /// Create a `Mamba1Config` from a `UniversalConfig`'s SSM section.
    pub fn from_universal(
        config: &crate::model::config::UniversalConfig,
    ) -> crate::error::Result<Self> {
        let ssm = config
            .ssm
            .as_ref()
            .ok_or_else(|| crate::error::Error::ModelError {
                reason: "Mamba1 requires ssm config section".into(),
            })?;
        if !ssm.is_mamba1() {
            return Err(crate::error::Error::ModelError {
                reason: format!(
                    "Mamba1 requires ssm.variant = 'mamba1', got '{}'",
                    ssm.variant
                ),
            });
        }
        Ok(Self {
            d_model: config.hidden_size,
            d_state: ssm.state_size,
            expand: ssm.expand,
            d_conv: ssm.conv_kernel,
            chunk_size: ssm.chunk_size,
            dt_softplus: true,
            use_d: true,
            hidden_state_clamp: Some(DEFAULT_HIDDEN_STATE_CLAMP),
        })
    }

    /// Inner (expanded) dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Total projected dimension for `in_proj` output.
    pub fn in_proj_dim(&self) -> usize {
        2 * self.d_inner()
    }

    /// Total projected dimension for `x_proj` output: dt, B, C.
    pub fn x_proj_dim(&self) -> usize {
        self.d_inner() + 2 * self.d_state
    }

    /// Depthwise conv channels.
    pub fn conv_channels(&self) -> usize {
        self.d_inner()
    }

    /// Shared sequential scan config: represent each Mamba1 inner channel as one SSM head.
    pub fn scan_config(&self) -> Mamba2Config {
        Mamba2Config {
            d_model: self.d_model,
            d_state: self.d_state,
            nheads: self.d_inner(),
            headdim: 1,
            expand: self.expand,
            ngroups: 1,
            d_conv: self.d_conv,
            chunk_size: self.chunk_size,
            use_dt_bias: false,
            use_d: self.use_d,
            dt_softplus: false,
        }
    }

    /// Validate config constraints. Returns error if invalid.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(Error::ModelError {
                reason: "d_model must be > 0".to_string(),
            });
        }
        if self.d_state == 0 {
            return Err(Error::ModelError {
                reason: "d_state must be > 0".to_string(),
            });
        }
        if self.expand == 0 {
            return Err(Error::ModelError {
                reason: "expand must be > 0".to_string(),
            });
        }
        if self.d_conv == 0 {
            return Err(Error::ModelError {
                reason: "d_conv must be > 0".to_string(),
            });
        }
        if self.chunk_size == 0 {
            return Err(Error::ModelError {
                reason: "chunk_size must be > 0".to_string(),
            });
        }
        if let Some(clamp) = self.hidden_state_clamp.filter(|clamp| *clamp <= 0.0) {
            return Err(Error::ModelError {
                reason: format!("hidden_state_clamp must be > 0, got {clamp}"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba1_config_defaults() {
        let config = Mamba1Config::new(256);
        assert_eq!(config.d_model, 256);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.expand, 2);
        assert_eq!(config.d_inner(), 512);
        assert_eq!(config.d_conv, 4);
        assert!(config.dt_softplus);
        assert!(config.use_d);
        assert_eq!(config.hidden_state_clamp, Some(DEFAULT_HIDDEN_STATE_CLAMP));
    }

    #[test]
    fn test_mamba1_config_builders() {
        let config = Mamba1Config::new(256)
            .with_d_state(32)
            .with_expand(3)
            .with_d_conv(5)
            .with_chunk_size(8)
            .with_dt_softplus(false)
            .with_use_d(false)
            .with_hidden_state_clamp(None);
        assert_eq!(config.d_state, 32);
        assert_eq!(config.expand, 3);
        assert_eq!(config.d_inner(), 768);
        assert_eq!(config.d_conv, 5);
        assert_eq!(config.chunk_size, 8);
        assert!(!config.dt_softplus);
        assert!(!config.use_d);
        assert_eq!(config.hidden_state_clamp, None);
    }

    #[test]
    fn test_mamba1_config_validation() {
        assert!(Mamba1Config::new(256).validate().is_ok());
        assert!(Mamba1Config::new(256).with_d_state(0).validate().is_err());
        assert!(Mamba1Config::new(256).with_expand(0).validate().is_err());
        assert!(Mamba1Config::new(256).with_d_conv(0).validate().is_err());
        assert!(
            Mamba1Config::new(256)
                .with_hidden_state_clamp(Some(0.0))
                .validate()
                .is_err()
        );
    }
}

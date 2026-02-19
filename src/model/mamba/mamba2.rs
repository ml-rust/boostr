//! Mamba2 layer implementing SSD (Structured State Space Duality)
//!
//! Architecture: in_proj → causal conv1d → SiLU → SSM → gate → out_proj
//!
//! SSM algorithms (sequential scan, chunked SSD) live in the `ssm` module.

use crate::error::{Error, Result};
use crate::model::mamba::ssm::{SsmInput, var_contiguous};
use crate::nn::{Conv1d, Linear, RmsNorm, VarBuilder};
use numr::autograd::{
    Var, var_add, var_exp, var_mul, var_narrow, var_neg, var_reshape, var_silu, var_softplus,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, ConvOps, NormalizationOps, PaddingMode, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

/// Mamba2 layer implementing the SSD algorithm.
///
/// Forward: in_proj → split(z, xBC, dt) → conv1d(xBC) → silu → split(x,B,C) →
///          SSM(x, A, B, C, dt, D) → gate(silu(z)) → norm? → out_proj
pub struct Mamba2<R: Runtime> {
    config: Mamba2Config,
    in_proj: Linear<R>,
    conv1d: Conv1d<R>,
    out_proj: Linear<R>,
    a_log: Var<R>,
    dt_bias: Option<Var<R>>,
    d_param: Option<Var<R>>,
    norm: Option<RmsNorm<R>>,
}

/// Bundled weight tensors for Mamba2 construction (reduces parameter count).
pub struct Mamba2Weights<R: Runtime> {
    pub in_proj: Linear<R>,
    pub conv1d: Conv1d<R>,
    pub out_proj: Linear<R>,
    pub a_log: Tensor<R>,
    pub dt_bias: Option<Tensor<R>>,
    pub d_param: Option<Tensor<R>>,
    pub norm: Option<RmsNorm<R>>,
}

impl<R: Runtime> Mamba2<R> {
    /// Create a new Mamba2 layer from config and weights.
    pub fn new(config: Mamba2Config, weights: Mamba2Weights<R>, trainable: bool) -> Self {
        Self {
            config,
            in_proj: weights.in_proj,
            conv1d: weights.conv1d,
            out_proj: weights.out_proj,
            a_log: Var::new(weights.a_log, trainable),
            dt_bias: weights.dt_bias.map(|t| Var::new(t, trainable)),
            d_param: weights.d_param.map(|t| Var::new(t, trainable)),
            norm: weights.norm,
        }
    }

    /// Load from a VarBuilder (HuggingFace Mamba2 naming).
    pub fn from_varbuilder(
        config: &Mamba2Config,
        vb: &mut VarBuilder<R>,
        trainable: bool,
    ) -> Result<Self> {
        config.validate()?;
        let conv_channels = config.conv_channels();

        let in_proj_weight = vb.take_tensor("in_proj.weight")?;
        let in_proj_bias = if vb.contains("in_proj.bias") {
            Some(vb.take_tensor("in_proj.bias")?)
        } else {
            None
        };
        let in_proj = Linear::new(in_proj_weight, in_proj_bias, trainable);

        let conv_weight = vb.take_tensor("conv1d.weight")?;
        let conv_bias = if vb.contains("conv1d.bias") {
            Some(vb.take_tensor("conv1d.bias")?)
        } else {
            None
        };
        let causal_pad = config.d_conv - 1;
        let conv1d = Conv1d::new(
            conv_weight,
            conv_bias,
            1,
            PaddingMode::Custom(causal_pad, 0, 0, 0),
            1,
            conv_channels,
            trainable,
        );

        let out_proj_weight = vb.take_tensor("out_proj.weight")?;
        let out_proj_bias = if vb.contains("out_proj.bias") {
            Some(vb.take_tensor("out_proj.bias")?)
        } else {
            None
        };
        let out_proj = Linear::new(out_proj_weight, out_proj_bias, trainable);

        let a_log = vb.take_tensor("A_log")?;
        let dt_bias = if config.use_dt_bias && vb.contains("dt_bias") {
            Some(vb.take_tensor("dt_bias")?)
        } else {
            None
        };
        let d_param = if config.use_d && vb.contains("D") {
            Some(vb.take_tensor("D")?)
        } else {
            None
        };
        let norm = if vb.contains("norm.weight") {
            let norm_weight = vb.take_tensor("norm.weight")?;
            Some(RmsNorm::new(norm_weight, 1e-5, trainable))
        } else {
            None
        };

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias,
            d_param,
            norm,
        };
        Ok(Self::new(config.clone(), weights, trainable))
    }

    /// Forward pass.
    ///
    /// x: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ActivationOps<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Error::ModelError {
                reason: format!("expected [batch, seq_len, d_model], got shape {:?}", shape),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        if shape[2] != self.config.d_model {
            return Err(Error::ModelError {
                reason: format!(
                    "d_model mismatch: expected {}, got {}",
                    self.config.d_model, shape[2]
                ),
            });
        }

        let d_inner = self.config.d_inner();
        let n_groups_d_state = self.config.ngroups * self.config.d_state;

        // 1. Input projection: [B, S, d_model] -> [B, S, proj_dim]
        let projected = self.in_proj.forward(client, x)?;

        // 2. Split into z, xBC, dt
        let xbc_len = d_inner + 2 * n_groups_d_state;
        let z = var_contiguous(&var_narrow(&projected, 2, 0, d_inner).map_err(Error::Numr)?);
        let xbc =
            var_contiguous(&var_narrow(&projected, 2, d_inner, xbc_len).map_err(Error::Numr)?);
        let dt = var_contiguous(
            &var_narrow(&projected, 2, d_inner + xbc_len, self.config.nheads)
                .map_err(Error::Numr)?,
        );

        // 3. Causal conv1d on xBC
        let xbc_ncl = var_contiguous(&var_transpose(&xbc).map_err(Error::Numr)?);
        let xbc_conv = self.conv1d.forward(client, xbc_ncl.tensor())?;
        let xbc_conv_var = Var::new(xbc_conv, false);
        let xbc = var_contiguous(&var_transpose(&xbc_conv_var).map_err(Error::Numr)?);

        // 4. SiLU activation
        let xbc = var_silu(&xbc, client).map_err(Error::Numr)?;

        // 5. Split xBC into x_ssm, B, C
        let x_ssm = var_contiguous(&var_narrow(&xbc, 2, 0, d_inner).map_err(Error::Numr)?);
        let b_proj =
            var_contiguous(&var_narrow(&xbc, 2, d_inner, n_groups_d_state).map_err(Error::Numr)?);
        let c_proj = var_contiguous(
            &var_narrow(&xbc, 2, d_inner + n_groups_d_state, n_groups_d_state)
                .map_err(Error::Numr)?,
        );

        // 6. Reshape for SSM
        let x_ssm = var_reshape(
            &x_ssm,
            &[batch, seq_len, self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)?;
        let b_proj = var_reshape(
            &b_proj,
            &[batch, seq_len, self.config.ngroups, self.config.d_state],
        )
        .map_err(Error::Numr)?;
        let c_proj = var_reshape(
            &c_proj,
            &[batch, seq_len, self.config.ngroups, self.config.d_state],
        )
        .map_err(Error::Numr)?;

        // 7. Compute A = -exp(A_log)
        let a = var_neg(&var_exp(&self.a_log, client).map_err(Error::Numr)?, client)
            .map_err(Error::Numr)?;

        // 8. Process dt
        let mut dt = dt;
        if self.config.dt_softplus {
            dt = var_softplus(&dt, client).map_err(Error::Numr)?;
        }
        if let Some(ref bias) = self.dt_bias {
            dt = var_add(&dt, bias, client).map_err(Error::Numr)?;
        }

        // 9. SSM forward
        let ssm_input = SsmInput {
            x: &x_ssm,
            a: &a,
            b: &b_proj,
            c: &c_proj,
            d_param: self.d_param.as_ref(),
            dt: &dt,
            config: &self.config,
        };
        let out = crate::model::mamba::ssm::ssm_forward_sequential(client, &ssm_input)?;

        // 10. Reshape back: [B, S, nheads, headdim] -> [B, S, d_inner]
        let out = var_reshape(&out, &[batch, seq_len, d_inner]).map_err(Error::Numr)?;

        // 11. Gate: out = out * silu(z)
        let z_gate = var_silu(&z, client).map_err(Error::Numr)?;
        let out = var_mul(&out, &z_gate, client).map_err(Error::Numr)?;

        // 12. Optional norm
        let out = if let Some(ref norm) = self.norm {
            norm.forward(client, &out)?
        } else {
            out
        };

        // 13. Output projection
        self.out_proj.forward(client, &out)
    }

    pub fn config(&self) -> &Mamba2Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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

    #[test]
    fn test_mamba2_forward_shape() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(8)
            .with_nheads(1)
            .with_d_state(4)
            .with_expand(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);

        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let proj_dim = config.proj_dim();

        let in_proj_w = Tensor::<CpuRuntime>::from_slice(
            &[0.01f32; 328], // 8 * 41 = 328
            &[proj_dim, 8],
            &device,
        );
        let in_proj = Linear::new(in_proj_w, None, false);

        let conv_w = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32; 96], // 24 * 4 = 96
            &[conv_channels, 1, 4],
            &device,
        );
        let conv1d = Conv1d::new(
            conv_w,
            None,
            1,
            PaddingMode::Custom(3, 0, 0, 0),
            1,
            conv_channels,
            false,
        );

        let out_proj_w = Tensor::<CpuRuntime>::from_slice(
            &[0.01f32; 128], // 16 * 8 = 128
            &[8, d_inner],
            &device,
        );
        let out_proj = Linear::new(out_proj_w, None, false);

        let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[config.nheads], &device);

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias: None,
            d_param: None,
            norm: None,
        };
        let mamba = Mamba2::new(config, weights, false);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[1, 4, 8], &device),
            false,
        );

        let out = mamba.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_mamba2_forward_invalid_input() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(8)
            .with_nheads(1)
            .with_d_state(4)
            .with_expand(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);

        let d_inner = config.d_inner();
        let conv_channels = config.conv_channels();
        let proj_dim = config.proj_dim();

        let in_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 328], &[proj_dim, 8], &device),
            None,
            false,
        );
        let conv1d = Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 96], &[conv_channels, 1, 4], &device),
            None,
            1,
            PaddingMode::Custom(3, 0, 0, 0),
            1,
            conv_channels,
            false,
        );
        let out_proj = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&[0.01f32; 128], &[8, d_inner], &device),
            None,
            false,
        );
        let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[1], &device);

        let weights = Mamba2Weights {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            dt_bias: None,
            d_param: None,
            norm: None,
        };
        let mamba = Mamba2::new(config, weights, false);

        // 2D input should fail
        let x_2d = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[1, 8], &device),
            false,
        );
        assert!(mamba.forward(&client, &x_2d).is_err());

        // Wrong d_model should fail
        let x_wrong = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[1, 4, 3], &device),
            false,
        );
        assert!(mamba.forward(&client, &x_wrong).is_err());
    }
}

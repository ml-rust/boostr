//! Gated DeltaNet state: per-layer recurrent state for `qwen35` inference.
//!
//! Analogous to [`SsmState`](crate::inference::SsmState) for Mamba2. Stores:
//!
//! - Conv window: `[batch, qkv_dim, conv_kernel - 1]`, the last inputs of
//!   the depthwise causal conv (see `nn::causal_conv1d`)
//! - Recurrence state: `[batch, value_heads, S, S]` in the
//!   `GatedDeltaNetOps` orientation (row = key index, column = value index)

use crate::error::Result;
use crate::model::config::GdnConfig;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One GDN layer's recurrent state.
pub struct GdnState<R: Runtime> {
    /// Conv window: `[batch, qkv_dim, conv_kernel - 1]`.
    conv: Tensor<R>,
    /// Delta-rule state: `[batch, value_heads, S, S]`.
    ssm: Tensor<R>,
    /// True once a forward pass wrote the state.
    initialized: bool,
}

impl<R: Runtime<DType = DType>> GdnState<R> {
    /// Zero state for one layer.
    pub fn zeros(cfg: &GdnConfig, batch: usize, dtype: DType, device: &R::Device) -> Result<Self> {
        let conv = Tensor::<R>::zeros(&[batch, cfg.qkv_dim(), cfg.conv_kernel - 1], dtype, device)?;
        let ssm = Tensor::<R>::zeros(
            &[batch, cfg.value_heads, cfg.state_size, cfg.state_size],
            dtype,
            device,
        )?;
        Ok(Self {
            conv,
            ssm,
            initialized: false,
        })
    }

    /// Conv window: `[batch, qkv_dim, conv_kernel - 1]`.
    pub fn conv(&self) -> &Tensor<R> {
        &self.conv
    }

    /// Delta-rule state: `[batch, value_heads, S, S]`.
    pub fn ssm(&self) -> &Tensor<R> {
        &self.ssm
    }

    /// Batch size the state was built for.
    pub fn batch(&self) -> usize {
        self.conv.shape()[0]
    }

    /// True once a forward pass wrote the state.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Replace both tensors after a forward pass.
    pub fn update(&mut self, conv: Tensor<R>, ssm: Tensor<R>) {
        self.conv = conv;
        self.ssm = ssm;
        self.initialized = true;
    }

    /// Zero both tensors and clear `initialized`.
    pub fn reset(&mut self) -> Result<()> {
        let device = self.ssm.device().clone();
        let dtype = self.ssm.dtype();
        self.conv = Tensor::<R>::zeros(self.conv.shape(), dtype, &device)?;
        self.ssm = Tensor::<R>::zeros(self.ssm.shape(), dtype, &device)?;
        self.initialized = false;
        Ok(())
    }
}

/// Per-layer GDN states for a whole model, indexed by GDN layer index
/// (the position among the model's GDN layers, not the absolute layer index).
pub struct LayeredGdnState<R: Runtime> {
    layers: Vec<GdnState<R>>,
}

impl<R: Runtime<DType = DType>> LayeredGdnState<R> {
    /// Zero states for `num_layers` GDN layers.
    pub fn zeros(
        num_layers: usize,
        cfg: &GdnConfig,
        batch: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|_| GdnState::zeros(cfg, batch, dtype, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// Mutable state of GDN layer `idx`.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut GdnState<R>> {
        self.layers.get_mut(idx)
    }

    /// State of GDN layer `idx`.
    pub fn layer(&self, idx: usize) -> Option<&GdnState<R>> {
        self.layers.get(idx)
    }

    /// GDN layer count.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Reset every layer.
    pub fn reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn cfg() -> GdnConfig {
        GdnConfig {
            hidden_size: 8,
            conv_kernel: 4,
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            inner_size: 16,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: false,
        }
    }

    #[test]
    fn zeros_have_layout_shapes() {
        let device = CpuDevice::new();
        let state = GdnState::<CpuRuntime>::zeros(&cfg(), 2, DType::F32, &device).unwrap();
        assert_eq!(state.conv().shape(), &[2, 8 + 8 + 16, 3]);
        assert_eq!(state.ssm().shape(), &[2, 4, 4, 4]);
        assert_eq!(state.batch(), 2);
        assert!(!state.is_initialized());
    }

    #[test]
    fn update_then_reset() {
        let device = CpuDevice::new();
        let mut state = GdnState::<CpuRuntime>::zeros(&cfg(), 1, DType::F32, &device).unwrap();
        let conv = Tensor::<CpuRuntime>::ones(&[1, 32, 3], DType::F32, &device).unwrap();
        let ssm = Tensor::<CpuRuntime>::ones(&[1, 4, 4, 4], DType::F32, &device).unwrap();
        state.update(conv, ssm);
        assert!(state.is_initialized());
        assert_eq!(state.ssm().to_vec::<f32>()[0], 1.0);

        state.reset().unwrap();
        assert!(!state.is_initialized());
        assert_eq!(state.ssm().to_vec::<f32>()[0], 0.0);
        assert_eq!(state.conv().shape(), &[1, 32, 3]);
    }

    #[test]
    fn layered_indexing() {
        let device = CpuDevice::new();
        let layers =
            LayeredGdnState::<CpuRuntime>::zeros(3, &cfg(), 1, DType::F32, &device).unwrap();
        assert_eq!(layers.num_layers(), 3);
        assert!(layers.layer(2).is_some());
        assert!(layers.layer(3).is_none());
    }
}

//! SSM State Cache — per-layer recurrent state for Mamba2 inference
//!
//! Analogous to `KvCache` for transformers. Stores:
//! - SSM hidden state `h`: `[batch, nheads, headdim, d_state]`
//! - Conv buffer: `[batch, conv_channels, d_conv - 1]` (sliding window for causal conv1d)

use crate::model::mamba::mamba2::Mamba2Config;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-layer SSM state for Mamba2 inference.
pub struct SsmState<R: Runtime> {
    /// SSM hidden state: `[batch, nheads, headdim, d_state]`
    h: Tensor<R>,
    /// Conv sliding window buffer: `[batch, conv_channels, d_conv - 1]`
    conv_state: Tensor<R>,
    /// Whether any tokens have been processed (state is valid)
    initialized: bool,
}

impl<R: Runtime<DType = DType>> SsmState<R> {
    /// Create a new zeroed SSM state for a single layer.
    pub fn new(batch_size: usize, config: &Mamba2Config, dtype: DType, device: &R::Device) -> Self {
        let h = Tensor::<R>::zeros(
            &[batch_size, config.nheads, config.headdim, config.d_state],
            dtype,
            device,
        );
        let conv_state = Tensor::<R>::zeros(
            &[batch_size, config.conv_channels(), config.d_conv - 1],
            dtype,
            device,
        );
        Self {
            h,
            conv_state,
            initialized: false,
        }
    }

    /// Get the SSM hidden state.
    pub fn h(&self) -> &Tensor<R> {
        &self.h
    }

    /// Get the conv sliding window buffer.
    pub fn conv_state(&self) -> &Tensor<R> {
        &self.conv_state
    }

    /// Whether state has been initialized by at least one forward pass.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Update SSM hidden state after a forward pass.
    pub fn update_h(&mut self, new_h: Tensor<R>) {
        self.h = new_h;
        self.initialized = true;
    }

    /// Update conv buffer after a forward pass.
    pub fn update_conv_state(&mut self, new_conv: Tensor<R>) {
        self.conv_state = new_conv;
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        let shape = self.h.shape().to_vec();
        let dtype = self.h.dtype();
        let conv_shape = self.conv_state.shape().to_vec();
        let device = self.h.device().clone();
        self.h = Tensor::<R>::zeros(&shape, dtype, &device);
        self.conv_state = Tensor::<R>::zeros(&conv_shape, dtype, &device);
        self.initialized = false;
    }
}

/// Multi-layer SSM state cache for a full Mamba2 model.
pub struct LayeredSsmState<R: Runtime> {
    layers: Vec<SsmState<R>>,
}

impl<R: Runtime<DType = DType>> LayeredSsmState<R> {
    /// Create a new multi-layer SSM state.
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        config: &Mamba2Config,
        dtype: DType,
        device: &R::Device,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| SsmState::new(batch_size, config, dtype, device))
            .collect();
        Self { layers }
    }

    /// Get mutable reference to a layer's state.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut SsmState<R>> {
        self.layers.get_mut(idx)
    }

    /// Get reference to a layer's state.
    pub fn layer(&self, idx: usize) -> Option<&SsmState<R>> {
        self.layers.get(idx)
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_ssm_state_create() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let config = Mamba2Config::new(64)
            .with_nheads(2)
            .with_d_state(16)
            .with_expand(2);

        let state = SsmState::<CpuRuntime>::new(1, &config, DType::F32, &device);
        assert_eq!(state.h().shape(), &[1, 2, 64, 16]);
        assert_eq!(state.conv_state().shape(), &[1, config.conv_channels(), 3]);
        assert!(!state.is_initialized());
    }

    #[test]
    fn test_layered_ssm_state() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let config = Mamba2Config::new(64)
            .with_nheads(2)
            .with_d_state(16)
            .with_expand(2);

        let state = LayeredSsmState::<CpuRuntime>::new(4, 1, &config, DType::F32, &device);
        assert_eq!(state.num_layers(), 4);
        assert!(state.layer(0).is_some());
        assert!(state.layer(4).is_none());
    }

    #[test]
    fn test_ssm_state_reset() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let config = Mamba2Config::new(64)
            .with_nheads(2)
            .with_d_state(16)
            .with_expand(2);

        let mut state = SsmState::<CpuRuntime>::new(1, &config, DType::F32, &device);
        let dummy_h = Tensor::<CpuRuntime>::ones(&[1, 2, 64, 16], DType::F32, &device);
        state.update_h(dummy_h);
        assert!(state.is_initialized());

        state.reset();
        assert!(!state.is_initialized());
    }
}

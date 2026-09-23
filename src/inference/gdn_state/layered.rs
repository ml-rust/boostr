//! Every GDN layer's state for one model.

use crate::error::Result;
use crate::inference::gdn_state::GdnState;
use crate::model::config::GdnConfig;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;

/// Per-layer GDN states for a whole model, indexed by GDN layer index
/// (the position among the model's GDN layers, not the absolute layer index).
pub struct LayeredGdnState<R: Runtime> {
    pub(super) layers: Vec<GdnState<R>>,
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

    /// Reset every layer in place; every buffer keeps its address.
    pub fn reset<C: BinaryOps<R>>(&mut self, client: &C) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset(client)?;
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
    fn layered_indexing() {
        let device = CpuDevice::new();
        let layers =
            LayeredGdnState::<CpuRuntime>::zeros(3, &cfg(), 1, DType::F32, &device).unwrap();
        assert_eq!(layers.num_layers(), 3);
        assert!(layers.layer(2).is_some());
        assert!(layers.layer(3).is_none());
    }
}

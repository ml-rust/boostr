//! Device mapping for model offloading
//!
//! Supports splitting model layers across GPU and CPU.

/// Device placement for a layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePlacement {
    /// Layer on GPU
    Gpu,
    /// Layer on CPU
    Cpu,
}

/// Maps model layers to devices for offloading
#[derive(Debug, Clone)]
pub struct LayerDeviceMap {
    /// Placement for each layer
    placements: Vec<DevicePlacement>,
    /// Embedding placement
    pub embed_placement: DevicePlacement,
    /// LM head placement
    pub lm_head_placement: DevicePlacement,
}

impl LayerDeviceMap {
    /// Create a device map with all layers on GPU
    pub fn all_gpu(num_layers: usize) -> Self {
        Self {
            placements: vec![DevicePlacement::Gpu; num_layers],
            embed_placement: DevicePlacement::Gpu,
            lm_head_placement: DevicePlacement::Gpu,
        }
    }

    /// Create a device map with specified number of GPU layers (rest on CPU)
    pub fn with_gpu_layers(num_layers: usize, gpu_layers: usize) -> Self {
        let mut placements = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i < gpu_layers {
                placements.push(DevicePlacement::Gpu);
            } else {
                placements.push(DevicePlacement::Cpu);
            }
        }
        Self {
            placements,
            embed_placement: DevicePlacement::Gpu,
            lm_head_placement: DevicePlacement::Gpu,
        }
    }

    /// Get placement for a specific layer
    pub fn placement(&self, layer_idx: usize) -> DevicePlacement {
        self.placements
            .get(layer_idx)
            .copied()
            .unwrap_or(DevicePlacement::Cpu)
    }

    /// Number of GPU layers
    pub fn gpu_layer_count(&self) -> usize {
        self.placements
            .iter()
            .filter(|p| **p == DevicePlacement::Gpu)
            .count()
    }

    /// Number of CPU layers
    pub fn cpu_layer_count(&self) -> usize {
        self.placements
            .iter()
            .filter(|p| **p == DevicePlacement::Cpu)
            .count()
    }
}

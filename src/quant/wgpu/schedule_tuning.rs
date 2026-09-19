//! WebGPU implementation of `ScheduleTuning`.

use crate::quant::QuantFormat;
use crate::quant::traits::ScheduleTuning;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl ScheduleTuning<WgpuRuntime> for WgpuClient {
    /// No schedule choice is measured on the WebGPU backend, so this is a
    /// no-op.
    fn warm_schedule_tuning(&self, _formats: &[QuantFormat]) {}
}

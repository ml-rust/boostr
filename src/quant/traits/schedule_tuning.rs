//! Per-device schedule-tuning warm-up trait.

use crate::quant::QuantFormat;
use numr::runtime::Runtime;

/// Pre-touch a client's per-device schedule-tuning cache for `formats`.
///
/// Every backend client implements this so generic model and engine code
/// calls it without knowing which backend runs under `R`. Only the CUDA
/// backend measures anything: a schedule choice there is a launch-time pick
/// between two bit-identical alternatives, cached per (device, key) through
/// `numr::runtime::cuda::tune::tuned`. CPU and WebGPU have no such choice,
/// so their implementations are no-ops.
pub trait ScheduleTuning<R: Runtime> {
    /// Warm every schedule-tuning key a forward pass over `formats` can
    /// reach on this client's device.
    ///
    /// Call once after the model loads, before the first CUDA graph
    /// capture: a capture refuses to probe, so an unwarmed key would
    /// silently keep its one-part fallback instead of this device's
    /// measured pick. Idempotent — a key already cached is read, not
    /// re-measured.
    fn warm_schedule_tuning(&self, formats: &[QuantFormat]);
}

//! Pre-touch of the per-device MMQ schedule-tuning keys, so a `tuned()`
//! lookup made later inside a CUDA graph capture hits a warm cache.
//!
//! `numr::runtime::cuda::tune::tuned` returns its fallback without probing
//! or caching when `client`'s stream is inside a capture — a probe records
//! events and synchronizes, which would invalidate the capture. Call
//! [`warm_schedule_tuning`] once per device after the model loads and
//! before the first capture; blazr's warmup does this. Every probe it runs
//! is idempotent: a key already in the tune cache is read, not re-measured,
//! so calling it more than once, or on a device the checkpoint's formats
//! turn out not to need, costs one cache read and nothing else.

use std::collections::HashSet;

use numr::runtime::cuda::CudaClient;
use numr::runtime::{Device, RuntimeClient};

use crate::quant::QuantFormat;

use super::format_dispatch::feat_major_format;
use super::mmq_feat_major::prefers_tile_parallel;

/// Touch every schedule-tuning key a forward pass over `formats` can reach
/// on `client`'s device: the small-M wave bound, once, and each format's
/// tile-parallel pick.
///
/// `k = 0` in the feature-major lookup is a whole number of every format's
/// block, so only the format and the device's int8 MMA cap gate it — the
/// same descriptor a real `k` resolves to. A format with no feature-major
/// kernel on this device, or a device without tuning enabled, is skipped:
/// `tuned`'s own fallback path already covers a probe failure, so there is
/// nothing left to pre-touch for it.
pub fn warm_schedule_tuning(client: &CudaClient, formats: &[QuantFormat]) {
    numr::runtime::cuda::kernels::smallm_max_waves(client);

    let device_index = client.device().id();
    let mut seen: HashSet<QuantFormat> = HashSet::new();
    for &format in formats {
        if !seen.insert(format) {
            continue;
        }
        if let Some(fm) = feat_major_format(format, 0, device_index) {
            prefers_tile_parallel(client, fm);
        }
    }
}

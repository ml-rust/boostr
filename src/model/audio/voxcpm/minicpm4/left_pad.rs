//! [`LeftPad`]: a left-padded batch's per-row start, on the device for the
//! attention kernels and on the host for the per-row RoPE positions.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-row left padding of a batch: row `b`'s first real position is
/// `starts[b]`, so it holds no key below it and its RoPE positions count
/// from there.
///
/// Both copies describe the same values. `starts` feeds the flash kernels'
/// `kv_start`; `host` lets the attention block rotate each row at its own
/// positions without a device round trip per layer. A padded row then
/// forms the same floats its own unpadded run forms: the same cos/sin rows,
/// the same keys in the same tiles.
pub struct LeftPad<R: Runtime> {
    /// `[B]` I32 on the device.
    pub starts: Tensor<R>,
    /// The same starts on the host.
    pub host: Vec<i32>,
}

impl<R: Runtime<DType = DType>> LeftPad<R> {
    /// Uploads `starts` and keeps the host copy. Errors on a negative start.
    pub fn new(starts: Vec<i32>, device: &R::Device) -> Result<Self> {
        if let Some(bad) = starts.iter().find(|&&s| s < 0) {
            return Err(Error::InvalidArgument {
                arg: "starts",
                reason: format!("left pad {bad} is negative"),
            });
        }
        let device_starts = Tensor::<R>::from_slice(&starts, &[starts.len()], device)?;
        Ok(Self {
            starts: device_starts,
            host: starts,
        })
    }

    /// Rows in the batch.
    pub fn batch(&self) -> usize {
        self.host.len()
    }

    /// Whether any row is padded at all. An all-zero pad is the unpadded
    /// call, and the callers take the unpadded path for it.
    pub fn is_padded(&self) -> bool {
        self.host.iter().any(|&s| s > 0)
    }

    /// Row `b`'s start, as a count of positions.
    pub fn start(&self, b: usize) -> usize {
        usize::try_from(self.host[b].max(0)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn keeps_both_copies_and_reports_padding() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 3], &device).expect("pad");
        assert_eq!(pad.batch(), 2);
        assert!(pad.is_padded());
        assert_eq!(pad.start(1), 3);
        assert_eq!(pad.starts.to_vec::<i32>(), vec![0, 3]);
        let none = LeftPad::<CpuRuntime>::new(vec![0, 0], &device).expect("pad");
        assert!(!none.is_padded());
    }

    #[test]
    fn rejects_a_negative_start() {
        let device = CpuDevice::new();
        assert!(LeftPad::<CpuRuntime>::new(vec![0, -1], &device).is_err());
    }
}

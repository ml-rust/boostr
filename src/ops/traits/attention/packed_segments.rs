//! Segment offsets for packed (variable-length) attention.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Segment boundaries of a packed `[total_tokens, H, D]` buffer.
///
/// Holds the cumulative offsets twice: once on the host, once as an I32
/// device tensor. The constructor builds both from one host slice, so they
/// always agree. Kernels read the device copy. Host-side loops read
/// `host_cu_seqlens`, so no backend reads offsets back from the device.
///
/// Segment `i` covers tokens `host_cu_seqlens[i]..host_cu_seqlens[i + 1]`.
/// Attention never crosses a segment boundary.
#[derive(Clone)]
pub struct PackedSegments<R: Runtime> {
    cu_seqlens: Tensor<R>,
    host_cu_seqlens: Vec<i32>,
    max_seqlen: usize,
    num_segments: usize,
}

impl<R: Runtime<DType = DType>> PackedSegments<R> {
    /// Build from cumulative offsets and upload them to `device` once.
    ///
    /// `offsets` must start at 0, never decrease, and end at `total_tokens`.
    /// It must hold at least two entries, so at least one segment exists.
    /// Empty segments are allowed.
    pub fn from_offsets(offsets: &[i32], total_tokens: usize, device: &R::Device) -> Result<Self> {
        let max_seqlen = validate_cu_seqlens("offsets", offsets, total_tokens)?;
        let cu_seqlens = Tensor::<R>::from_slice(offsets, &[offsets.len()], device)?;
        Ok(Self {
            cu_seqlens,
            host_cu_seqlens: offsets.to_vec(),
            max_seqlen,
            num_segments: offsets.len() - 1,
        })
    }

    /// Build from per-segment token counts and upload the offsets once.
    ///
    /// `lengths` must hold at least one entry. The token total must fit in `i32`.
    pub fn from_lengths(lengths: &[usize], device: &R::Device) -> Result<Self> {
        let mut offsets = Vec::with_capacity(lengths.len() + 1);
        offsets.push(0i32);
        let mut total: i32 = 0;
        for (i, &len) in lengths.iter().enumerate() {
            total = i32::try_from(len)
                .ok()
                .and_then(|l| total.checked_add(l))
                .ok_or_else(|| Error::InvalidArgument {
                    arg: "lengths",
                    reason: format!(
                        "PackedSegments: token total overflows i32 at segment {i} \
                         (length {len}); split the batch into smaller packs"
                    ),
                })?;
            offsets.push(total);
        }
        Self::from_offsets(&offsets, total as usize, device)
    }
}

impl<R: Runtime> PackedSegments<R> {
    /// Cumulative offsets as a `[num_segments + 1]` I32 device tensor.
    pub fn cu_seqlens(&self) -> &Tensor<R> {
        &self.cu_seqlens
    }

    /// Host copy of the cumulative offsets.
    pub fn host_cu_seqlens(&self) -> &[i32] {
        &self.host_cu_seqlens
    }

    /// Token count of the longest segment.
    pub fn max_seqlen(&self) -> usize {
        self.max_seqlen
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    /// Total packed token count: the last offset.
    pub fn total_tokens(&self) -> usize {
        // The constructor checks the slice holds at least two non-negative entries.
        self.host_cu_seqlens.last().map_or(0, |&t| t as usize)
    }
}

/// Check one cumulative-offset table and return its longest segment.
///
/// `offsets` must hold at least two entries, start at 0, never decrease, and
/// end at `total_tokens`. Backends call this on the host offsets they receive.
pub(crate) fn validate_cu_seqlens(
    arg: &'static str,
    offsets: &[i32],
    total_tokens: usize,
) -> Result<usize> {
    let invalid = |reason: String| Error::InvalidArgument { arg, reason };
    if offsets.len() < 2 {
        return Err(invalid(format!(
            "cu_seqlens needs at least 2 entries (one segment), got {}; \
             pass [0, total_tokens] for a single segment",
            offsets.len()
        )));
    }
    if offsets[0] != 0 {
        return Err(invalid(format!(
            "cu_seqlens must start at 0, got {}; offsets are cumulative token counts",
            offsets[0]
        )));
    }
    let mut max_seqlen = 0usize;
    for (i, pair) in offsets.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(invalid(format!(
                "cu_seqlens must not decrease: entry {} is {} after {}",
                i + 1,
                pair[1],
                pair[0]
            )));
        }
        max_seqlen = max_seqlen.max((pair[1] - pair[0]) as usize);
    }
    let last = offsets[offsets.len() - 1] as usize;
    if last != total_tokens {
        return Err(invalid(format!(
            "cu_seqlens ends at {last}, but the packed buffer holds {total_tokens} tokens; \
             the last offset must equal the token count"
        )));
    }
    Ok(max_seqlen)
}

/// Check host offsets passed to a varlen backend against its `batch_size`.
pub(crate) fn validate_host_cu_seqlens(
    arg: &'static str,
    offsets: &[i32],
    batch_size: usize,
    total_tokens: usize,
) -> Result<()> {
    if offsets.len() != batch_size + 1 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "host cu_seqlens holds {} entries, batch_size {batch_size} needs {}",
                offsets.len(),
                batch_size + 1
            ),
        });
    }
    validate_cu_seqlens(arg, offsets, total_tokens).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn reason(err: Error) -> String {
        match err {
            Error::InvalidArgument { reason, .. } => reason,
            other => format!("unexpected error kind: {other}"),
        }
    }

    #[test]
    fn from_offsets_uploads_matching_copies() {
        let (_, dev) = cpu_setup();
        let segs = PackedSegments::<CpuRuntime>::from_offsets(&[0, 3, 3, 8], 8, &dev).unwrap();
        assert_eq!(segs.num_segments(), 3);
        assert_eq!(segs.max_seqlen(), 5);
        assert_eq!(segs.total_tokens(), 8);
        assert_eq!(segs.host_cu_seqlens(), &[0, 3, 3, 8]);
        assert_eq!(segs.cu_seqlens().to_vec::<i32>(), vec![0, 3, 3, 8]);
        assert_eq!(segs.cu_seqlens().dtype(), numr::dtype::DType::I32);
    }

    #[test]
    fn from_lengths_builds_cumulative_offsets() {
        let (_, dev) = cpu_setup();
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[4, 2, 6], &dev).unwrap();
        assert_eq!(segs.host_cu_seqlens(), &[0, 4, 6, 12]);
        assert_eq!(segs.max_seqlen(), 6);
        assert_eq!(segs.total_tokens(), 12);
    }

    #[test]
    fn rejects_empty_and_single_entry() {
        let (_, dev) = cpu_setup();
        let e = PackedSegments::<CpuRuntime>::from_offsets(&[], 0, &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("at least 2 entries"));
        let e = PackedSegments::<CpuRuntime>::from_offsets(&[0], 0, &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("at least 2 entries"));
        let e = PackedSegments::<CpuRuntime>::from_lengths(&[], &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("at least 2 entries"));
    }

    #[test]
    fn rejects_nonzero_start() {
        let (_, dev) = cpu_setup();
        let e = PackedSegments::<CpuRuntime>::from_offsets(&[1, 4], 4, &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("must start at 0"));
    }

    #[test]
    fn rejects_decreasing_offsets() {
        let (_, dev) = cpu_setup();
        let e = PackedSegments::<CpuRuntime>::from_offsets(&[0, 5, 3, 6], 6, &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("must not decrease"));
    }

    #[test]
    fn rejects_wrong_total() {
        let (_, dev) = cpu_setup();
        let e = PackedSegments::<CpuRuntime>::from_offsets(&[0, 3, 6], 7, &dev)
            .err()
            .unwrap();
        let r = reason(e);
        assert!(r.contains("ends at 6") && r.contains("7 tokens"), "{r}");
    }

    #[test]
    fn rejects_i32_overflow() {
        let (_, dev) = cpu_setup();
        let big = i32::MAX as usize;
        let e = PackedSegments::<CpuRuntime>::from_lengths(&[big, 1], &dev)
            .err()
            .unwrap();
        assert!(reason(e).contains("overflows i32"));
    }

    #[test]
    fn host_check_rejects_batch_mismatch() {
        let e = validate_host_cu_seqlens("host_cu_seqlens_q", &[0, 2, 4], 3, 4)
            .err()
            .unwrap();
        assert!(reason(e).contains("batch_size 3 needs 4"));
        assert!(validate_host_cu_seqlens("host_cu_seqlens_q", &[0, 2, 4], 2, 4).is_ok());
    }
}

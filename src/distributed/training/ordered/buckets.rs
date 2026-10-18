//! Bucket plans, stream ordering and errors shared by the ordered collectives.

use std::ops::Range;

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::{Communicator, Runtime, RuntimeClient};

/// Default bucket size of the ordered collectives: 64 MiB.
pub const DEFAULT_ORDERED_BUCKET_BYTES: usize = 64 * 1024 * 1024;

/// One run of consecutive segments that travels as one flat buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BucketPlan {
    /// Indices of the segments in this bucket.
    pub(crate) segments: Range<usize>,
    /// Element dtype of every segment in this bucket.
    pub(crate) dtype: DType,
    /// Total element count of this bucket.
    pub(crate) elems: usize,
}

/// Group consecutive segments into buckets of at most `bucket_bytes`.
///
/// A segment is `(elements, dtype)`. A dtype change starts a new bucket.
/// A segment larger than `bucket_bytes` gets a bucket of its own.
/// The plan depends only on the segment list. Equal lists give equal plans on every rank.
pub(crate) fn plan_buckets(segments: &[(usize, DType)], bucket_bytes: usize) -> Vec<BucketPlan> {
    let mut plans: Vec<BucketPlan> = Vec::new();
    for (index, &(elems, dtype)) in segments.iter().enumerate() {
        let elem_bytes = dtype.size_in_bytes();
        let fits = plans.last().is_some_and(|last| {
            last.dtype == dtype
                && last.elems.saturating_add(elems).saturating_mul(elem_bytes) <= bucket_bytes
        });
        match plans.last_mut() {
            Some(last) if fits => {
                last.segments.end = index + 1;
                last.elems += elems;
            }
            _ => plans.push(BucketPlan {
                segments: index..index + 1,
                dtype,
                elems,
            }),
        }
    }
    plans
}

/// Order the communicator's stream after all compute work queued so far.
///
/// NCCL runs collectives on its own stream. An event makes that stream wait for the
/// compute stream, so a collective never reads a buffer before its writer finishes.
/// Without stream events, the host waits for the client instead.
pub(crate) fn order_after_compute<R: Runtime, C: RuntimeClient<R>>(
    comm: &dyn Communicator,
    client: &C,
) -> Result<()> {
    let (Some(sync), Some(stream)) = (comm.as_stream_sync(), client.compute_stream_handle()) else {
        client.synchronize();
        return Ok(());
    };
    let event = sync
        .create_event()
        .map_err(|e| dist_err(format!("create event before the collectives failed: {e}")))?;
    let ordered = sync
        .record_on_stream(event, stream)
        .and_then(|()| sync.comm_stream_wait_event(event));
    let destroyed = sync.destroy_event(event);
    ordered.map_err(|e| {
        dist_err(format!(
            "order the comm stream after the compute stream failed: {e}"
        ))
    })?;
    destroyed.map_err(|e| dist_err(format!("destroy event {event} failed: {e}")))
}

/// Wait for every collective that `caller` issued on `comm`.
pub(crate) fn sync_comm(comm: &dyn Communicator, caller: &str) -> Result<()> {
    comm.sync()
        .map_err(|e| dist_err(format!("{caller}: sync after the collectives failed: {e}")))
}

/// Build a distributed error from `reason`.
pub(crate) fn dist_err(reason: String) -> Error {
    Error::DistributedError { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(segments: Range<usize>, dtype: DType, elems: usize) -> BucketPlan {
        BucketPlan {
            segments,
            dtype,
            elems,
        }
    }

    #[test]
    fn test_empty_list_gives_no_buckets() {
        assert!(plan_buckets(&[], 1024).is_empty());
    }

    #[test]
    fn test_segments_pack_until_the_bucket_is_full() {
        let segments = [(2, DType::F32), (2, DType::F32), (2, DType::F32)];
        let plans = plan_buckets(&segments, 16);
        assert_eq!(
            plans,
            vec![plan(0..2, DType::F32, 4), plan(2..3, DType::F32, 2)]
        );
    }

    #[test]
    fn test_dtype_change_starts_a_bucket() {
        let segments = [
            (1, DType::F32),
            (1, DType::I64),
            (1, DType::I64),
            (1, DType::F32),
        ];
        let plans = plan_buckets(&segments, 1024);
        assert_eq!(
            plans,
            vec![
                plan(0..1, DType::F32, 1),
                plan(1..3, DType::I64, 2),
                plan(3..4, DType::F32, 1),
            ]
        );
    }

    #[test]
    fn test_oversized_segment_gets_its_own_bucket() {
        let segments = [(1, DType::F32), (100, DType::F32), (1, DType::F32)];
        let plans = plan_buckets(&segments, 8);
        assert_eq!(
            plans,
            vec![
                plan(0..1, DType::F32, 1),
                plan(1..2, DType::F32, 100),
                plan(2..3, DType::F32, 1),
            ]
        );
    }

    #[test]
    fn test_equal_lists_give_equal_plans() {
        let segments = [(3, DType::F32), (5, DType::BF16), (7, DType::BF16)];
        assert_eq!(plan_buckets(&segments, 12), plan_buckets(&segments, 12));
    }
}

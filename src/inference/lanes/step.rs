//! The KV range and write position a lane-parallel decode step reads.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One decode step's view of a batched KV cache, a lane per sequence.
///
/// Lane `b` holds its own sequence from cache position `kv_start[b]`, writes
/// this step's token at `write_pos[b]`, and attends over
/// `[kv_start[b], kv_end[b])`. `kv_end[b]` is therefore one past
/// `write_pos[b]`, and a lane with no sequence in it carries
/// `kv_end[b] == kv_start[b]`, an empty range.
///
/// Both tensors are `[batch]` I32 on the cache's device and keep their
/// addresses across steps, so a captured graph can hold them.
///
/// `write_pos` stays a host slice rather than a third device tensor: the KV
/// write walks it lane by lane on the host (`KvCache::update_lanes` narrows
/// and writes one lane at a time), so the position is never needed on
/// device and a device round trip to read it back would be wasted work.
pub struct LaneKvStep<'a, R: Runtime> {
    /// `[batch]` I32: lane `b`'s first cache position.
    pub kv_start: &'a Tensor<R>,
    /// `[batch]` I32: one past lane `b`'s last cache position.
    pub kv_end: &'a Tensor<R>,
    /// Lane `b`'s write position, on the host: no device read decides it.
    pub write_pos: &'a [usize],
}

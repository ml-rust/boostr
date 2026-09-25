//! [`CacheStep`]: what one KV-cached attention call writes and which keys
//! each row of the batch then reads.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::audio::voxcpm::minicpm4::left_pad::LeftPad;
use crate::ops::traits::{KvCacheOps, KvSeqLen};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Where this call's K/V lands, row by row.
enum Write<'a, R: Runtime> {
    /// Every row writes the same run of cache slots, starting at this
    /// position. The prefill's whole prefix and any decode whose rows all
    /// sit at one position.
    Fused(usize),
    /// Row `b` writes one position at `pos[b]`. `kv_end[b]` is one past its
    /// last key, a `[B]` I32 tensor on the cache's device.
    PerLane {
        pos: &'a [usize],
        kv_end: &'a Tensor<R>,
    },
}

/// One cached attention call's per-row write position and key range.
///
/// Row `b` holds no key below `pad.starts[b]`, writes this call's K/V from
/// [`row_write`](Self::row_write), and rotates at
/// `row_write(b) - row_start(b)`. A batch whose rows all sit at one
/// position and carry no padding collapses to the single-position call the
/// unpadded path has always made — see [`uniform_position`](Self::uniform_position)
/// and [`kv_seq_len`](Self::kv_seq_len).
pub struct CacheStep<'a, R: Runtime> {
    pad: Option<&'a LeftPad<R>>,
    write: Write<'a, R>,
}

impl<'a, R: Runtime<DType = DType>> CacheStep<'a, R> {
    /// A run of `seq` positions starting at `position` in every row, over a
    /// batch left-padded by `pad`.
    pub fn prefix(position: usize, pad: Option<&'a LeftPad<R>>) -> Self {
        Self {
            pad,
            write: Write::Fused(position),
        }
    }

    /// One position per row: row `b` writes at `pos[b]` and ends at
    /// `kv_end[b]`, over a batch left-padded by `pad`.
    pub fn lanes(pad: Option<&'a LeftPad<R>>, pos: &'a [usize], kv_end: &'a Tensor<R>) -> Self {
        Self {
            pad,
            write: Write::PerLane { pos, kv_end },
        }
    }

    /// Row `b`'s first cache position, `0` when the batch is unpadded.
    pub fn row_start(&self, b: usize) -> usize {
        match self.pad {
            Some(pad) if b < pad.batch() => pad.start(b),
            _ => 0,
        }
    }

    /// The first cache slot row `b` writes on this call.
    pub fn row_write(&self, b: usize) -> usize {
        match &self.write {
            Write::Fused(position) => *position,
            Write::PerLane { pos, .. } => pos.get(b).copied().unwrap_or(0),
        }
    }

    /// The furthest first-write across the rows, which is the slot the
    /// cache's own cursor tracks.
    pub fn last_write(&self) -> usize {
        match &self.write {
            Write::Fused(position) => *position,
            Write::PerLane { pos, .. } => pos.iter().copied().max().unwrap_or(0),
        }
    }

    /// The `[B]` I32 first cache position the flash kernels take, or `None`
    /// for a batch that carries no padding tensor at all.
    pub fn kv_start(&self) -> Option<&'a Tensor<R>> {
        self.pad.map(|pad| &pad.starts)
    }

    /// The one absolute position every row rotates and attends at, or
    /// `None` when the rows disagree.
    ///
    /// `Some` is the batch-wide RoPE call: one `apply_rope` over every row
    /// against one slice of the table. `None` splits the rotation row by
    /// row, which forms a different float sequence, so this collapses to
    /// `Some` whenever the rows genuinely agree.
    pub fn uniform_position(&self) -> Option<usize> {
        if self.pad.is_some_and(LeftPad::is_padded) {
            return None;
        }
        match &self.write {
            Write::Fused(position) => Some(*position),
            Write::PerLane { pos, .. } => uniform(pos),
        }
    }

    /// The key bound the flash kernel takes: `Shared` while every row ends
    /// at the same key, `PerRow` once they differ.
    ///
    /// The `Shared` arm is not an optimization. CUDA's flash dispatch
    /// refuses `PerRow` at every shape but `S_q == 1`, and on CPU a
    /// `PerRow` bound moves an unpadded call off `standard_attention_fwd`
    /// onto the per-row composed path — the same math in a different
    /// reduction order. So `PerRow` is emitted only when the ends genuinely
    /// differ, which is the only case it describes something `Shared`
    /// cannot.
    pub fn kv_seq_len(&self, cache_seq_len: usize) -> KvSeqLen<'a, R> {
        match &self.write {
            Write::Fused(_) => KvSeqLen::Shared(cache_seq_len),
            Write::PerLane { pos, kv_end } => match uniform(pos) {
                Some(_) => KvSeqLen::Shared(cache_seq_len),
                None => KvSeqLen::PerRow(*kv_end),
            },
        }
    }

    /// Write this call's post-RoPE K and V into `cache`.
    ///
    /// A fused write puts every row at the cache's own cursor in one
    /// launch. A per-lane write walks the rows and puts each one at its own
    /// position, which is the same bytes in the same slots whenever the
    /// positions agree.
    ///
    /// # Errors
    ///
    /// Propagates the cache's own refusal: a write past `max_seq_len`, a
    /// row count that does not match the cache's batch, or a non-contiguous
    /// `k`/`v` on the per-lane path.
    pub fn write_kv<C: KvCacheOps<R>>(
        &self,
        cache: &mut KvCache<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        client: &C,
    ) -> Result<()>
    where
        R::Client: IndexingOps<R>,
    {
        match &self.write {
            Write::Fused(_) => cache.update_fused(k, v, client),
            Write::PerLane { pos, .. } => cache.update_lanes(k, v, pos, client),
        }
    }

    /// Refuse a write that is not the slot the cache will actually fill.
    ///
    /// A query rotated for one absolute position and filed at another stays
    /// shape-valid and silently computes a different model, so this is
    /// checked before any tensor is touched. Per lane: a row never writes
    /// below its own first cache position, and never past the cache's
    /// filled length. Across lanes: the furthest write is exactly that
    /// filled length, so the longest row appends rather than overwrites.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the row and both positions.
    pub fn check_write_order(&self, filled: usize) -> Result<()> {
        match &self.write {
            Write::Fused(position) => {
                if *position != filled {
                    return Err(Error::InvalidArgument {
                        arg: "position",
                        reason: format!(
                            "expected {filled} (the next free cache slot, kv_cache.seq_len()), \
                             got {position}; positions must be written in order from 0"
                        ),
                    });
                }
                Ok(())
            }
            Write::PerLane { pos, .. } => {
                for (b, &p) in pos.iter().enumerate() {
                    let start = self.row_start(b);
                    if p < start {
                        return Err(Error::InvalidArgument {
                            arg: "position",
                            reason: format!(
                                "row {b} writes at {p}, below its own first cache position \
                                 {start}; a row's write position is its own filled length"
                            ),
                        });
                    }
                    if p > filled {
                        return Err(Error::InvalidArgument {
                            arg: "position",
                            reason: format!(
                                "row {b} writes at {p}, past the next free cache slot \
                                 {filled} (kv_cache.seq_len()); a row's write position is \
                                 its own filled length"
                            ),
                        });
                    }
                }
                let furthest = self.last_write();
                if furthest != filled {
                    return Err(Error::InvalidArgument {
                        arg: "position",
                        reason: format!(
                            "the furthest row writes at {furthest} but the next free cache \
                             slot is {filled} (kv_cache.seq_len()); the longest row appends \
                             on every step"
                        ),
                    });
                }
                Ok(())
            }
        }
    }
}

/// The one value every entry holds, or `None` when they differ or there are
/// none.
fn uniform(pos: &[usize]) -> Option<usize> {
    let first = *pos.first()?;
    pos.iter().all(|p| *p == first).then_some(first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn ends(values: &[i32], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, &[values.len()], device).expect("ends")
    }

    fn is_shared(len: &KvSeqLen<'_, CpuRuntime>) -> bool {
        matches!(len, KvSeqLen::Shared(_))
    }

    /// Rows that all sit at one position take the shared bound; one row
    /// behind the others takes the per-row tensor.
    #[test]
    fn shared_until_the_row_ends_differ() {
        let device = CpuDevice::new();
        let kv_end = ends(&[5, 5], &device);
        let (level_pos, ragged_pos) = ([4usize, 4], [4usize, 2]);

        let level = CacheStep::<CpuRuntime>::lanes(None, &level_pos, &kv_end);
        assert!(is_shared(&level.kv_seq_len(4)));
        assert_eq!(level.uniform_position(), Some(4));

        let ragged = CacheStep::<CpuRuntime>::lanes(None, &ragged_pos, &kv_end);
        assert!(!is_shared(&ragged.kv_seq_len(4)));
        assert_eq!(ragged.uniform_position(), None);
    }

    /// A prefill always takes the shared bound, padded or not: every row
    /// writes the same run of slots, and the padding moves each row's start
    /// rather than its end.
    #[test]
    fn a_prefix_write_is_always_shared() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");

        let padded = CacheStep::prefix(0, Some(&pad));
        assert!(is_shared(&padded.kv_seq_len(5)));
        // A padded batch rotates row by row, so there is no one position.
        assert_eq!(padded.uniform_position(), None);
        assert_eq!(padded.row_start(1), 2);

        let plain = CacheStep::<CpuRuntime>::prefix(3, None);
        assert!(is_shared(&plain.kv_seq_len(3)));
        assert_eq!(plain.uniform_position(), Some(3));
        assert_eq!(plain.row_start(1), 0);
    }

    /// A padded batch whose rows all sit at one position still ends at one
    /// key, so the bound stays shared while the rotation splits per row.
    #[test]
    fn padding_splits_the_rotation_but_not_the_bound() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");
        let kv_end = ends(&[6, 6], &device);
        let pos = [5usize, 5];
        let step = CacheStep::lanes(Some(&pad), &pos, &kv_end);

        assert!(is_shared(&step.kv_seq_len(5)));
        assert_eq!(step.uniform_position(), None);
        assert_eq!(step.kv_start().map(Tensor::shape), Some(&[2][..]));
    }

    /// The per-lane guard accepts a row sitting at its own filled length
    /// and rejects one that writes below its start, past the cache, or
    /// leaves the longest row short of it.
    #[test]
    fn the_per_lane_guard_rejects_a_mismatched_write_position() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");
        let kv_end = ends(&[6, 6], &device);

        CacheStep::lanes(Some(&pad), &[5, 3], &kv_end)
            .check_write_order(5)
            .expect("row 1 is behind but inside its own range");

        let below = CacheStep::lanes(Some(&pad), &[5, 1], &kv_end)
            .check_write_order(5)
            .expect_err("row 1 writes below its own first cache position");
        assert!(below.to_string().contains("row 1"), "{below}");

        let past = CacheStep::lanes(Some(&pad), &[7, 3], &kv_end)
            .check_write_order(5)
            .expect_err("row 0 writes past the cache");
        assert!(past.to_string().contains("row 0"), "{past}");

        let short = CacheStep::lanes(Some(&pad), &[4, 3], &kv_end)
            .check_write_order(5)
            .expect_err("no row appends");
        assert!(short.to_string().contains("furthest"), "{short}");
    }

    /// A fused write still names `kv_cache.seq_len()` and the slot it
    /// disagrees with.
    #[test]
    fn the_fused_guard_names_the_slot_it_wanted() {
        let err = CacheStep::<CpuRuntime>::prefix(9, None)
            .check_write_order(1)
            .expect_err("9 is not the next free slot");
        let text = err.to_string();
        assert!(text.contains('9'), "{text}");
        assert!(text.contains("seq_len"), "{text}");
        assert!(text.contains("next free cache slot"), "{text}");
        CacheStep::<CpuRuntime>::prefix(1, None)
            .check_write_order(1)
            .expect("1 is the next free slot");
    }
}

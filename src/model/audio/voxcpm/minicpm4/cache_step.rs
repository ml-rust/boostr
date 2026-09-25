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
    /// last key, a `[B]` I32 tensor on the cache's device. `occupied[b]`
    /// says whether lane `b` holds a sequence at all.
    PerLane {
        pos: &'a [usize],
        kv_end: &'a Tensor<R>,
        occupied: &'a [bool],
    },
}

/// What a KV cache already holds, as [`CacheStep::check_write_order`] reads
/// it.
///
/// `filled` is the cache's own cursor — the slot a FUSED write appends at —
/// and `lanes` is one past each lane's last key. The two agree only while
/// every lane holds the same run; a cache filled one sequence per lane makes
/// `filled` their maximum, which is no lane's next slot.
pub struct CacheFill<'a> {
    /// `KvCache::seq_len`: where a fused write lands.
    pub filled: usize,
    /// `KvCache::lane_lens`: one past each lane's last key.
    pub lanes: &'a [usize],
    /// Cache positions this cache can hold at all (`max_seq_len`).
    pub limit: usize,
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
    /// `kv_end[b]`, over a batch left-padded by `pad`. `occupied[b]` is
    /// `false` for a lane holding no sequence.
    pub fn lanes(
        pad: Option<&'a LeftPad<R>>,
        pos: &'a [usize],
        kv_end: &'a Tensor<R>,
        occupied: &'a [bool],
    ) -> Self {
        Self {
            pad,
            write: Write::PerLane {
                pos,
                kv_end,
                occupied,
            },
        }
    }

    /// Whether lane `b` holds a sequence. A fused write covers every row, so
    /// there is no free lane on that arm.
    fn is_occupied(&self, b: usize) -> bool {
        match &self.write {
            Write::Fused(_) => true,
            Write::PerLane { occupied, .. } => occupied.get(b).copied().unwrap_or(true),
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
            Write::PerLane { pos, .. } => self.uniform_occupied(pos),
        }
    }

    /// The one position every OCCUPIED lane sits at, or `None` when they
    /// differ or none is occupied.
    ///
    /// A free lane sits at its own position 0 and holds nothing, so counting
    /// it would call a batch ragged for a row that has no sequence to be
    /// ragged about — and one live row beside free lanes is the commonest
    /// batch there is.
    fn uniform_occupied(&self, pos: &[usize]) -> Option<usize> {
        let mut seated = pos
            .iter()
            .enumerate()
            .filter(|(b, _)| self.is_occupied(*b))
            .map(|(_, p)| *p);
        let first = seated.next()?;
        seated.all(|p| p == first).then_some(first)
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
    ///
    /// `cache_seq_len` is the cache's own cursor and bounds a FUSED write
    /// alone, where every row ends exactly there. It is the MAXIMUM over
    /// lanes and only ever rises, so on the per-lane arm it is not any
    /// row's key count: once a long sequence has run in a lane, a shorter
    /// one seated in its place still leaves the cursor high, and bounding
    /// the softmax by it would attend over the slots the previous occupant
    /// wrote. A lane's keys run from cache position 0 to its own write
    /// position, and this call adds the one position it writes, so the
    /// shared bound is that position plus one.
    pub fn kv_seq_len(&self, cache_seq_len: usize) -> KvSeqLen<'a, R> {
        match &self.write {
            Write::Fused(_) => KvSeqLen::Shared(cache_seq_len),
            Write::PerLane { pos, kv_end, .. } => match self.uniform_occupied(pos) {
                Some(position) => KvSeqLen::Shared(position + 1),
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
    /// checked before any tensor is touched.
    ///
    /// A fused write covers every row at once, so it must land on the
    /// cache's own cursor. A per-lane write is checked LANE BY LANE against
    /// that lane's own end: an occupied lane writes exactly where its keys
    /// stop, never below its first cache position, and never past the
    /// positions the cache holds. This is what catches the two stacks
    /// drifting apart — they share one lane table and one advance, so a
    /// stack stepped once more than the other writes a position past its own
    /// keys and is refused here.
    ///
    /// The cache's cursor is NOT the per-lane rule. It is the maximum over
    /// lanes and only ever rises, so once a long sequence has run, a shorter
    /// one seated in its place would never reach it.
    ///
    /// A free lane is skipped: it holds no sequence, writes at its own
    /// position 0, and its empty key range keeps that write out of every
    /// softmax.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the row and both positions.
    pub fn check_write_order(&self, fill: &CacheFill<'_>) -> Result<()> {
        match &self.write {
            Write::Fused(position) => {
                if *position != fill.filled {
                    return Err(Error::InvalidArgument {
                        arg: "position",
                        reason: format!(
                            "expected {} (the next free cache slot, kv_cache.seq_len()), \
                             got {position}; positions must be written in order from 0",
                            fill.filled
                        ),
                    });
                }
                Ok(())
            }
            Write::PerLane { pos, .. } => {
                for (b, &p) in pos.iter().enumerate() {
                    if !self.is_occupied(b) {
                        continue;
                    }
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
                    let held = fill.lanes.get(b).copied().unwrap_or(fill.filled);
                    if p != held {
                        return Err(Error::InvalidArgument {
                            arg: "position",
                            reason: format!(
                                "row {b} writes at {p} but its own keys end at {held} \
                                 (kv_cache.lane_lens()); a row's write position is its own \
                                 filled length, so the two stacks sharing this lane table \
                                 have drifted apart"
                            ),
                        });
                    }
                    if p >= fill.limit {
                        return Err(Error::InvalidArgument {
                            arg: "position",
                            reason: format!(
                                "row {b} writes at {p}, past the {} cache positions this \
                                 cache holds (kv_cache.max_seq_len())",
                                fill.limit
                            ),
                        });
                    }
                }
                Ok(())
            }
        }
    }
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

    /// A cache whose lanes all end at `filled`, with room for `limit`.
    fn level_fill(filled: usize, lanes: &[usize], limit: usize) -> CacheFill<'_> {
        CacheFill {
            filled,
            lanes,
            limit,
        }
    }

    /// Every lane of a `batch`-row step, occupied.
    const HELD: [bool; 4] = [true; 4];

    /// Rows that all sit at one position take the shared bound; one row
    /// behind the others takes the per-row tensor.
    #[test]
    fn shared_until_the_row_ends_differ() {
        let device = CpuDevice::new();
        let kv_end = ends(&[5, 5], &device);
        let (level_pos, ragged_pos) = ([4usize, 4], [4usize, 2]);

        let level = CacheStep::<CpuRuntime>::lanes(None, &level_pos, &kv_end, &HELD);
        assert!(is_shared(&level.kv_seq_len(4)));
        assert_eq!(level.uniform_position(), Some(4));

        let ragged = CacheStep::<CpuRuntime>::lanes(None, &ragged_pos, &kv_end, &HELD);
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
        let step = CacheStep::lanes(Some(&pad), &pos, &kv_end, &HELD);

        assert!(is_shared(&step.kv_seq_len(5)));
        assert_eq!(step.uniform_position(), None);
        assert_eq!(step.kv_start().map(Tensor::shape), Some(&[2][..]));
    }

    /// Each lane is checked against its OWN end: a ragged pair both sitting
    /// where their keys stop is accepted, and a lane writing below its start,
    /// off its own end, or past the cache is refused.
    #[test]
    fn the_per_lane_guard_checks_each_lane_against_its_own_end() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");
        let kv_end = ends(&[6, 4], &device);

        CacheStep::lanes(Some(&pad), &[5, 3], &kv_end, &HELD)
            .check_write_order(&level_fill(5, &[5, 3], 8))
            .expect("each lane sits at its own end");

        let below = CacheStep::lanes(Some(&pad), &[5, 1], &kv_end, &HELD)
            .check_write_order(&level_fill(5, &[5, 1], 8))
            .expect_err("row 1 writes below its own first cache position");
        assert!(below.to_string().contains("row 1"), "{below}");

        let drifted = CacheStep::lanes(Some(&pad), &[5, 3], &kv_end, &HELD)
            .check_write_order(&level_fill(5, &[5, 4], 8))
            .expect_err("row 1 writes one behind its own keys");
        assert!(drifted.to_string().contains("row 1"), "{drifted}");
        assert!(drifted.to_string().contains("lane_lens"), "{drifted}");

        let past = CacheStep::lanes(Some(&pad), &[8, 3], &kv_end, &HELD)
            .check_write_order(&level_fill(8, &[8, 3], 8))
            .expect_err("row 0 writes past the cache");
        assert!(past.to_string().contains("row 0"), "{past}");
    }

    /// The cursor is the maximum over lanes and only rises, so a short lane
    /// seated where a long one ran is accepted on its own end — the case a
    /// cursor-relative rule refuses forever.
    #[test]
    fn a_reseated_short_lane_is_not_held_to_the_cursor() {
        let device = CpuDevice::new();
        let kv_end = ends(&[4, 901], &device);
        let occupied = [true, true];
        CacheStep::<CpuRuntime>::lanes(None, &[3, 900], &kv_end, &occupied)
            .check_write_order(&level_fill(900, &[3, 900], 1024))
            .expect("lane 0 was re-seated short and writes at its own end");
    }

    /// A free lane holds no sequence: it writes at its own position 0 and
    /// the guard passes over it whatever the cache's lane still holds.
    #[test]
    fn a_free_lane_is_not_checked() {
        let device = CpuDevice::new();
        let kv_end = ends(&[0, 7], &device);
        let occupied = [false, true];
        CacheStep::<CpuRuntime>::lanes(None, &[0, 6], &kv_end, &occupied)
            .check_write_order(&level_fill(900, &[900, 6], 1024))
            .expect("a free lane's stale cache end is not its next slot");
    }

    /// The shared bound on the per-lane arm is the rows' OWN key end, never
    /// the cache's cursor.
    ///
    /// This is the case a long request leaves behind: it drove the cursor
    /// high, its lane was freed, and short rows now sit far below it. A
    /// bound taken from the cursor would put the previous occupant's keys
    /// in every row's softmax, which no shape check and no write-order
    /// check can see.
    #[test]
    fn the_shared_bound_is_the_rows_own_end_not_the_cursor() {
        let device = CpuDevice::new();
        let kv_end = ends(&[4, 4], &device);
        let pos = [3usize, 3];
        let step = CacheStep::<CpuRuntime>::lanes(None, &pos, &kv_end, &HELD);

        // The cache's cursor still stands where a previous, longer occupant
        // of these lanes left it.
        match step.kv_seq_len(900) {
            KvSeqLen::Shared(len) => assert_eq!(
                len, 4,
                "rows holding 3 keys and writing a fourth must attend over 4"
            ),
            KvSeqLen::PerRow(_) => panic!("rows sitting at one position take the shared bound"),
        }
    }

    /// A free lane sits at position 0 and holds nothing, so it must not make
    /// an otherwise level batch ragged — one live row beside free lanes is
    /// the commonest batch a continuous engine steps.
    #[test]
    fn a_free_lane_does_not_make_the_batch_ragged() {
        let device = CpuDevice::new();
        let kv_end = ends(&[7, 0], &device);
        let occupied = [true, false];
        let pos = [6usize, 0];
        let step = CacheStep::<CpuRuntime>::lanes(None, &pos, &kv_end, &occupied);

        assert_eq!(step.uniform_position(), Some(6));
        match step.kv_seq_len(900) {
            KvSeqLen::Shared(len) => assert_eq!(len, 7),
            KvSeqLen::PerRow(_) => panic!("one occupied lane is uniform by itself"),
        }
    }

    /// A fused write still names `kv_cache.seq_len()` and the slot it
    /// disagrees with.
    #[test]
    fn the_fused_guard_names_the_slot_it_wanted() {
        let err = CacheStep::<CpuRuntime>::prefix(9, None)
            .check_write_order(&level_fill(1, &[1], 8))
            .expect_err("9 is not the next free slot");
        let text = err.to_string();
        assert!(text.contains('9'), "{text}");
        assert!(text.contains("seq_len"), "{text}");
        assert!(text.contains("next free cache slot"), "{text}");
        CacheStep::<CpuRuntime>::prefix(1, None)
            .check_write_order(&level_fill(1, &[1], 8))
            .expect("1 is the next free slot");
    }
}

//! Per-lane device scalars updated before each graph replay, and the shared
//! stream-ordered RoPE-slice copy both decode graphs use.

use cudarc::driver::sys;
use numr::ops::HostCopyOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// Device-resident decode scalars, one value per lane, updated before each
/// graph replay.
///
/// Both tensors are `[batch]` I32, allocated once and written in place
/// through [`HostCopyOps::write_host_slice`], so their device addresses hold
/// for the life of the struct and a captured graph can point at them. The
/// write is a host-to-device memcpy rather than `cuMemsetD32Async` because a
/// memset broadcasts one word to every element and cannot carry a different
/// value per lane.
///
/// Both host staging buffers live on the struct, so a step allocates
/// nothing.
pub struct DeviceScalars {
    /// `[batch]` I32 attention loop bound, one per lane.
    ///
    /// Passed to `decode_attention_*` as `seq_len_k_ptr`; the kernels
    /// dereference it at runtime, so the bound can differ between replays.
    pub seq_len_k: Tensor<CudaRuntime>,

    /// `[batch]` I32 KV insert position, one per lane.
    ///
    /// Passed to `kv_insert` as `write_pos_ptr`: lane `b` writes this step's
    /// K/V at `write_pos[b]` of the full-capacity cache.
    pub write_pos: Tensor<CudaRuntime>,

    /// Staging for `seq_len_k`, one entry per lane.
    ends: Vec<i32>,
    /// Staging for `write_pos`, one entry per lane.
    slots: Vec<i32>,
}

impl DeviceScalars {
    /// Allocate the scalars for `initial_seq_lens.len()` lanes, each holding
    /// its own initial length in both tensors.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `initial_seq_lens` is empty or a length
    /// is past what an I32 holds; an allocation error propagates.
    pub fn new(
        initial_seq_lens: &[usize],
        device: &numr::runtime::cuda::CudaDevice,
    ) -> Result<Self> {
        let batch = initial_seq_lens.len();
        if batch == 0 {
            return Err(Error::InferenceError {
                reason: "decode scalars need at least one lane, got 0 sequence lengths; \
                         allocate them with one length per lane the decode runs"
                    .into(),
            });
        }
        let mut vals = Vec::with_capacity(batch);
        for (lane, len) in initial_seq_lens.iter().enumerate() {
            vals.push(as_i32(*len, "seq_len", lane)?);
        }
        let seq_len_k = Tensor::<CudaRuntime>::from_slice(&vals, &[batch], device)?;
        let write_pos = Tensor::<CudaRuntime>::from_slice(&vals, &[batch], device)?;
        Ok(Self {
            seq_len_k,
            write_pos,
            ends: vec![0; batch],
            slots: vec![0; batch],
        })
    }

    /// Lane count both tensors hold.
    pub fn batch(&self) -> usize {
        self.slots.len()
    }

    /// Raw device pointer to the `[batch]` i32 seq_len_k buffer. Pass to
    /// decode_attention.
    pub fn seq_len_k_ptr(&self) -> u64 {
        self.seq_len_k.ptr()
    }

    /// Raw device pointer to the `[batch]` i32 write_pos buffer. Pass to
    /// kv_insert.
    pub fn write_pos_ptr(&self) -> u64 {
        self.write_pos.ptr()
    }

    /// Update `cos_slice` and `sin_slice` stable tensors with the rope values for `position`.
    ///
    /// Performs a stream-ordered D2D async copy of `half_dim` f32 elements from
    /// `rope_cos_cache[position * half_dim ..]` into `cos_slice` (and likewise for sin).
    /// Used to prepare RoPE values before each graph replay.
    #[allow(clippy::too_many_arguments)]
    pub fn update_rope_slices(
        &self,
        client: &CudaClient,
        rope_cos_cache: &Tensor<CudaRuntime>,
        rope_sin_cache: &Tensor<CudaRuntime>,
        cos_slice: &numr::autograd::Var<CudaRuntime>,
        sin_slice: &numr::autograd::Var<CudaRuntime>,
        position: usize,
        half_dim: usize,
    ) -> Result<()> {
        // Driver enqueues made outside numr's launch path hold the capture
        // permit themselves, scoped to the pair of copies and nothing else.
        let _permit = client.stream().enqueue_permit();
        let stream = client.stream().raw().cu_stream();
        copy_rope_slice_async(
            rope_cos_cache,
            position * half_dim,
            cos_slice.tensor(),
            half_dim,
            stream,
        )?;
        copy_rope_slice_async(
            rope_sin_cache,
            position * half_dim,
            sin_slice.tensor(),
            half_dim,
            stream,
        )?;
        Ok(())
    }

    /// Stream-ordered write of one sequence length per lane, for the decode
    /// step about to replay.
    ///
    /// `seq_lens[b]` is the number of tokens lane `b` holds before this
    /// step's insert, so:
    ///
    /// - `write_pos[b] = seq_lens[b]`   — where lane `b` inserts this step's K/V
    /// - `seq_len_k[b] = seq_lens[b] + 1` — how many K/V entries lane `b`
    ///   attends over AFTER the insert (positions `0..=seq_lens[b]`, matching
    ///   the non-graph path, where `update()` increments seq_len before
    ///   `get_kv()`)
    ///
    /// Both writes are host-to-device memcpies on the client's compute
    /// stream, ordered against the graph launch that follows, and they read
    /// the struct's own staging buffers, so the call allocates nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `seq_lens` does not hold one length per
    /// lane, or a length is past what an I32 holds; a copy error propagates.
    pub fn update(&mut self, client: &CudaClient, seq_lens: &[usize]) -> Result<()> {
        let batch = self.batch();
        if seq_lens.len() != batch {
            return Err(Error::InferenceError {
                reason: format!(
                    "decode scalars hold {batch} lanes but {} sequence lengths were given; pass \
                     one length per lane, or allocate the scalars with {} lanes",
                    seq_lens.len(),
                    seq_lens.len()
                ),
            });
        }

        self.ends.clear();
        self.slots.clear();
        for (lane, len) in seq_lens.iter().enumerate() {
            self.slots.push(as_i32(*len, "write_pos", lane)?);
            self.ends.push(as_i32(len + 1, "seq_len_k", lane)?);
        }

        client
            .write_host_slice(&self.write_pos, &self.slots)
            .map_err(Error::Numr)?;
        client
            .write_host_slice(&self.seq_len_k, &self.ends)
            .map_err(Error::Numr)
    }

    /// Stream-ordered write for a lane-parallel replay: one insert position
    /// per lane, and the shared key-axis clamp.
    ///
    /// `lens[b]` is the number of tokens lane `b` holds before this step's
    /// insert, so `write_pos[b] = lens[b]`, exactly as [`Self::update`]
    /// writes it. `seq_len_k` differs: the `_graph` kernels read it once per
    /// launch, from element 0, as the row-independent clamp on the key axis,
    /// while the lane table's `kv_end` carries the per-row bound. Every
    /// element carries `capacity`, the KV cache's static capacity, so the
    /// clamp never varies with a lane's live length.
    ///
    /// Both writes read the struct's own staging buffers, so the call
    /// allocates nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lens` does not hold one length per
    /// lane, or a length or `capacity` is past what an I32 holds; a copy
    /// error propagates.
    pub fn update_lanes(
        &mut self,
        client: &CudaClient,
        lens: &[usize],
        capacity: usize,
    ) -> Result<()> {
        let batch = self.batch();
        if lens.len() != batch {
            return Err(Error::InferenceError {
                reason: format!(
                    "decode scalars hold {batch} lanes but {} lane lengths were given; pass one \
                     length per lane, or allocate the scalars with {} lanes",
                    lens.len(),
                    lens.len()
                ),
            });
        }

        let bound = as_i32(capacity, "seq_len_k capacity", 0)?;

        self.slots.clear();
        for (lane, len) in lens.iter().enumerate() {
            self.slots.push(as_i32(*len, "write_pos", lane)?);
        }
        self.ends.clear();
        self.ends.resize(batch, bound);

        client
            .write_host_slice(&self.write_pos, &self.slots)
            .map_err(Error::Numr)?;
        client
            .write_host_slice(&self.seq_len_k, &self.ends)
            .map_err(Error::Numr)
    }
}

/// `value` as the I32 a device tensor holds, or an error naming the lane.
fn as_i32(value: usize, field: &'static str, lane: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InferenceError {
        reason: format!(
            "lane {lane}: {field} {value} is past what an I32 holds; restart the sequence with \
             a shorter prefix"
        ),
    })
}

/// Stream-ordered D2D async copy of `head_dim` elements from `src` at element
/// offset `src_elem_off` into `dst` starting at element 0.
///
/// `src` and `dst` share one dtype: a RoPE table is cast to the model dtype
/// at load, so the slice buffer must be allocated at the table's dtype.
///
/// Uses `cuMemcpyDtoDAsync_v2` so the copy is serialized on `stream` before any
/// subsequent stream operation (including `cuGraphLaunch`).
pub(super) fn copy_rope_slice_async(
    src: &Tensor<CudaRuntime>,
    src_elem_off: usize,
    dst: &Tensor<CudaRuntime>,
    head_dim: usize,
    stream: sys::CUstream,
) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(Error::InferenceError {
            reason: format!(
                "RoPE slice dtype {:?} != table dtype {:?}",
                dst.dtype(),
                src.dtype()
            ),
        });
    }
    let elem = src.dtype().size_in_bytes();
    let bytes = head_dim * elem;
    let src_ptr = src.ptr() + (src_elem_off * elem) as u64;
    let dst_ptr = dst.ptr();
    unsafe {
        let result = sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, stream);
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::InferenceError {
                reason: format!("cuMemcpyDtoDAsync_v2 for RoPE slice failed: {:?}", result),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;

    /// Every lane gets its own insert position and its own loop bound.
    #[test]
    fn a_multi_lane_update_lands_one_value_per_lane() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0, 0], &cuda.device).expect("scalars");

        scalars.update(&cuda.client, &[2, 7, 0]).expect("update");

        assert_eq!(scalars.batch(), 3);
        assert_eq!(scalars.write_pos.shape(), &[3]);
        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![2, 7, 0]);
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![3, 8, 1]);
    }

    /// One lane keeps the values the single-scalar path always wrote.
    #[test]
    fn one_lane_writes_the_slot_and_the_bound_past_it() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[5], &cuda.device).expect("scalars");

        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![5]);
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![5]);
        scalars.update(&cuda.client, &[5]).expect("update");
        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![5]);
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![6]);
    }

    /// A captured graph holds these addresses, so an update must not move
    /// either tensor.
    #[test]
    fn the_tensors_keep_their_addresses_across_updates() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0], &cuda.device).expect("scalars");
        let before = (scalars.seq_len_k_ptr(), scalars.write_pos_ptr());

        scalars.update(&cuda.client, &[1, 4]).expect("first");
        scalars.update(&cuda.client, &[9, 2]).expect("second");

        assert_eq!((scalars.seq_len_k_ptr(), scalars.write_pos_ptr()), before);
        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![9, 2]);
    }

    /// A length count that disagrees with the batch is refused, and the
    /// error names both.
    #[test]
    fn a_length_count_that_is_not_the_batch_is_refused() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0], &cuda.device).expect("scalars");

        let err = scalars
            .update(&cuda.client, &[1, 2, 3])
            .expect_err("three lengths for two lanes");

        let text = err.to_string();
        assert!(text.contains('2'), "{text}");
        assert!(text.contains('3'), "{text}");
    }

    /// A lane-parallel write keeps each lane's insert position and gives
    /// every lane the static capacity as the shared clamp, not a value
    /// derived from the lane lengths.
    #[test]
    fn a_lane_parallel_update_clamps_every_lane_at_the_capacity() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0, 0], &cuda.device).expect("scalars");

        scalars
            .update_lanes(&cuda.client, &[2, 7, 0], 64)
            .expect("write");

        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![2, 7, 0]);
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![64, 64, 64]);
    }

    /// The clamp is the capacity argument, not the longest lane: it does not
    /// move when a lane's length changes between calls.
    #[test]
    fn the_clamp_does_not_move_when_a_lanes_length_changes() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0], &cuda.device).expect("scalars");

        scalars
            .update_lanes(&cuda.client, &[1, 1], 32)
            .expect("first");
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![32, 32]);

        scalars
            .update_lanes(&cuda.client, &[30, 5], 32)
            .expect("second");
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![32, 32]);
    }

    /// One lane writes exactly what `update` writes when the capacity equals
    /// the length `update` would have bounded to.
    #[test]
    fn one_lane_reduces_to_the_single_lane_write() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0], &cuda.device).expect("scalars");

        scalars.update_lanes(&cuda.client, &[5], 6).expect("write");

        assert_eq!(scalars.write_pos.to_vec::<i32>(), vec![5]);
        assert_eq!(scalars.seq_len_k.to_vec::<i32>(), vec![6]);
    }

    /// A length count that disagrees with the batch is refused here too.
    #[test]
    fn a_lane_parallel_update_refuses_the_wrong_lane_count() {
        let Some(cuda) = cuda_setup() else { return };
        let mut scalars = DeviceScalars::new(&[0, 0], &cuda.device).expect("scalars");

        assert!(scalars.update_lanes(&cuda.client, &[1], 32).is_err());
    }

    /// A table with no lane has nothing to write.
    #[test]
    fn scalars_need_at_least_one_lane() {
        let Some(cuda) = cuda_setup() else { return };
        assert!(DeviceScalars::new(&[], &cuda.device).is_err());
    }
}

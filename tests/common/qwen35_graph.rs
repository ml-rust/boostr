//! Capturing and replaying a `qwen35` decode graph over a lane table.
//!
//! One capture serves a fixed lane count. The buffers every replay reads are
//! allocated here, before capture, so their device addresses hold.

use boostr::inference::decode_graph::{DeviceScalars, copy_into_stable};
use boostr::inference::{LaneTable, LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35Model;
use numr::dtype::DType;
use numr::runtime::CapturedGraph;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// The first capture arena is at most this fraction of free device memory.
const ARENA_FREE_DIVISOR: u64 = 4;
/// Upper bound for a capture arena.
const ARENA_CAP_BYTES: usize = 1 << 30;
/// Lower bound for a capture arena.
const ARENA_FLOOR_BYTES: usize = 1 << 20;

/// Bytes for a capture arena: a quarter of free device memory, capped.
///
/// One decode step records on the order of a thousand intermediate
/// allocations and the count scales with the lane count, so the arena is
/// probed from the device rather than fixed per batch. The capture reports
/// what it actually used.
pub fn arena_bytes(device: &CudaDevice) -> usize {
    let (free, _total) = device.memory_info().expect("device memory info");
    let quarter = usize::try_from(free / ARENA_FREE_DIVISOR).unwrap_or(ARENA_CAP_BYTES);
    quarter.clamp(ARENA_FLOOR_BYTES, ARENA_CAP_BYTES)
}

/// Stable-address buffers one captured `qwen35` decode step reads and
/// writes.
pub struct GraphBufs {
    /// `[batch, 1]` i64 embedding input, one token per lane.
    pub token_buf: Tensor<CudaRuntime>,
    /// `[batch, 1, vocab]` f32 copy of the step's logits, written by the
    /// captured copy and sampled on the host.
    pub logits_buf: Tensor<CudaRuntime>,
    /// Per-lane `write_pos` and the shared key bound.
    pub scalars: DeviceScalars,
}

impl GraphBufs {
    /// Allocate every buffer for `batch` lanes over a `vocab`-wide head.
    pub fn new(device: &CudaDevice, batch: usize, vocab: usize) -> Self {
        let lens = vec![0usize; batch];
        Self {
            token_buf: Tensor::<CudaRuntime>::zeros(&[batch, 1], DType::I64, device)
                .expect("token buf"),
            logits_buf: Tensor::<CudaRuntime>::zeros(&[batch, 1, vocab], DType::F32, device)
                .expect("logits buf"),
            scalars: DeviceScalars::new(&lens, device).expect("decode scalars"),
        }
    }

    /// Run the graph-mode forward once outside capture so the kernels JIT
    /// and the tuner probes.
    ///
    /// The pass writes the KV cache and the GDN buffers in place, so the
    /// caller restores both before capturing.
    pub fn warmup(
        &mut self,
        client: &CudaClient,
        model: &Qwen35Model<CudaRuntime>,
        kv: &LayeredKvCache<CudaRuntime>,
        gdn: &LayeredGdnState<CudaRuntime>,
        lanes: &LaneTable<CudaRuntime>,
        capacity: usize,
    ) {
        self.scalars
            .update_lanes(client, lanes.write_positions(), capacity)
            .expect("warmup scalars");
        let logits = model
            .forward_qwen35_graph_mode(client, &self.token_buf, kv, gdn, &self.scalars, lanes)
            .expect("warmup forward");
        copy_into_stable(client, &logits, &self.logits_buf).expect("warmup logits copy");
        let _ = self.logits_buf.to_vec::<f32>();
    }
}

/// Capture one decode step over `kv`, `gdn` and `lanes`, counting the
/// capture in `captures`.
///
/// Returns the graph and the arena footprint it reported, so a caller can
/// print what the step actually needed.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    kv: &LayeredKvCache<CudaRuntime>,
    gdn: &LayeredGdnState<CudaRuntime>,
    lanes: &LaneTable<CudaRuntime>,
    bufs: &GraphBufs,
    captures: &mut usize,
) -> (CapturedGraph<CudaRuntime>, usize) {
    let arena = arena_bytes(device);
    let batch = lanes.batch();
    let graph = CudaRuntime::capture_graph_into_with_arena(
        client,
        &[&bufs.token_buf],
        &[&bufs.logits_buf],
        arena,
        |c| {
            let logits = model
                .forward_qwen35_graph_mode(c, &bufs.token_buf, kv, gdn, &bufs.scalars, lanes)
                .map_err(|e| numr::error::Error::Backend(format!("capture forward: {e}")))?;
            copy_into_stable(c, &logits, &bufs.logits_buf)
        },
    )
    .unwrap_or_else(|e| {
        panic!(
            "capturing a {batch}-lane qwen35 decode step into a {arena} byte arena failed: {e}; \
             the error names the footprint the step needs, so raise the arena probe in \
             tests/common/qwen35_graph.rs to at least that many bytes"
        )
    });
    *captures += 1;
    let used = graph.arena_bytes_used().unwrap_or(0);
    println!("qwen35 decode graph: {batch} lanes, arena {arena}, used {used}");
    (graph, used)
}

/// Split a `[batch, 1, vocab]` logits readback into one row per lane.
pub fn lane_rows(all: &[f32], batch: usize, vocab: usize) -> Vec<Vec<f32>> {
    assert_eq!(all.len(), batch * vocab, "logits readback length");
    (0..batch)
        .map(|lane| all[lane * vocab..(lane + 1) * vocab].to_vec())
        .collect()
}

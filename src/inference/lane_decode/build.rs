//! Building a lane driver at a fixed lane count.
//!
//! Both arms take a batched KV cache and a batched GDN state already
//! allocated for `lanes` sequences at `capacity` tokens each, and allocate the
//! lane table and the `[lanes, 1]` token buffer once beside them.
//!
//! The graph arm adds an uncaptured warmup pass to JIT every kernel and to
//! shape the stable logits buffer, then captures one multi-lane decode step
//! into an arena. Every lane is claimed at length 0 for the warmup and the
//! capture, so the recorded kernels see one key per lane rather than an empty
//! range. The driver is reset before it is returned, so the session starts
//! with every lane free. The arena is sized generously first and tightened
//! from the reported footprint when the first guess is more than twice too
//! large.

use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{CapturedGraph, Runtime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::inference::decode_graph::{
    DeviceScalars, Qwen35DecodeGraph, arena_is_loose, copy_into_stable, first_arena_bytes,
    tight_arena_bytes,
};
use crate::inference::{LaneTable, LayeredGdnState, LayeredKvCache};
use crate::model::qwen35::Qwen35Model;

use super::driver::{EagerLanes, GraphLanes, LaneDecodeMode, LaneDriver};

/// Everything a lane driver is built from.
pub struct LaneDriverSpec<'a> {
    /// The model whose decode step every lane runs.
    pub model: &'a Qwen35Model<CudaRuntime>,
    /// Device the caches, the lane table and the graph live on.
    pub device: &'a CudaDevice,
    /// Batched KV cache, already allocated at `lanes` lanes.
    pub kv: LayeredKvCache<CudaRuntime>,
    /// Batched GDN state, already allocated at `lanes` rows.
    pub gdn: LayeredGdnState<CudaRuntime>,
    /// Sequences one step serves.
    pub lanes: usize,
    /// Tokens one lane holds at most.
    pub capacity: usize,
    /// Which decode path the driver steps.
    pub mode: LaneDecodeMode,
}

/// A built driver and the number of graph captures the build performed.
pub struct LaneDriverBuild {
    /// The driver, reset to an empty session.
    pub driver: LaneDriver,
    /// Captures the build ran: 0 on the eager arm, 1 or 2 on the graph arm.
    pub graph_captures: usize,
}

/// A decode driver for `spec.lanes` sequences of at most `spec.capacity`
/// tokens.
///
/// # Errors
///
/// [`Error::InferenceError`] when an allocation fails, or when the graph
/// arm's warmup or capture fails.
pub fn build_lane_driver(spec: LaneDriverSpec<'_>) -> Result<LaneDriverBuild> {
    match spec.mode {
        LaneDecodeMode::Eager => build_eager(spec),
        LaneDecodeMode::Graph => build_graph(spec),
    }
}

/// A lane table for `lanes` sequences, bounded by the model's own rope table.
fn lane_table(
    model: &Qwen35Model<CudaRuntime>,
    lanes: usize,
    device: &CudaDevice,
) -> Result<LaneTable<CudaRuntime>> {
    LaneTable::<CudaRuntime>::new(lanes, model.rope_table_len(), device)
}

/// The `[lanes, 1]` I64 buffer holding the token each lane decodes from.
fn token_buffer(lanes: usize, device: &CudaDevice) -> Result<Tensor<CudaRuntime>> {
    Ok(Tensor::<CudaRuntime>::zeros(
        &[lanes, 1],
        DType::I64,
        device,
    )?)
}

/// Batched caches, lane table and input buffer for the eager arm.
fn build_eager(spec: LaneDriverSpec<'_>) -> Result<LaneDriverBuild> {
    let table = lane_table(spec.model, spec.lanes, spec.device)?;
    let token_buf = token_buffer(spec.lanes, spec.device)?;
    Ok(LaneDriverBuild {
        driver: LaneDriver::Eager(Box::new(EagerLanes {
            lanes: table,
            kv: spec.kv,
            gdn: spec.gdn,
            token_buf,
        })),
        graph_captures: 0,
    })
}

/// Batched caches plus one decode graph captured at `spec.lanes` lanes.
fn build_graph(spec: LaneDriverSpec<'_>) -> Result<LaneDriverBuild> {
    let LaneDriverSpec {
        model,
        device,
        mut kv,
        mut gdn,
        lanes,
        capacity,
        mode: _,
    } = spec;
    let client = CudaRuntime::default_client(device);
    let mut table = lane_table(model, lanes, device)?;
    let token_buf = token_buffer(lanes, device)?;
    let initial_lens = vec![0usize; lanes];
    let mut scalars = DeviceScalars::new(&initial_lens, device)?;

    claim_every_lane(&client, &mut scalars, &mut table, capacity)?;

    let warmup = model
        .forward_qwen35_graph_mode(&client, &token_buf, &kv, &gdn, &scalars, &table)
        .map_err(|e| Error::InferenceError {
            reason: format!(
                "the {lanes}-lane graph-mode warmup forward failed: {e}; the capture needs one \
                 uncaptured pass to succeed first"
            ),
        })?;
    let logits_buf = Tensor::<CudaRuntime>::zeros(warmup.shape(), warmup.dtype(), device)?;
    copy_into_stable(&client, &warmup, &logits_buf)?;
    drop(warmup);

    // The warmup wrote both caches in place: clear them so capture records
    // against the same addresses holding no tokens.
    kv.reset();
    gdn.reset(&client)?;

    let bufs = CaptureBuffers {
        token_buf: &token_buf,
        logits_buf: &logits_buf,
        scalars: &scalars,
        table: &table,
        kv: &kv,
        gdn: &gdn,
    };
    let first_arena = first_arena_bytes(device)?;
    let mut graph = capture_step(model, &client, &bufs, first_arena)?;
    let used = graph.arena_bytes_used().unwrap_or(first_arena);
    let mut graph_captures = 1;
    tracing::info!(
        "{}-lane CUDA graph captured (arena={} used={})",
        lanes,
        first_arena,
        used
    );

    // Capture records without executing, so the caches still hold the same
    // state and serve a second capture unchanged.
    if arena_is_loose(used, first_arena) {
        let tight_arena = tight_arena_bytes(used);
        drop(graph);
        graph = capture_step(model, &client, &bufs, tight_arena)?;
        graph_captures += 1;
        tracing::info!(
            "{}-lane CUDA graph recaptured (arena={} used={:?})",
            lanes,
            tight_arena,
            graph.arena_bytes_used()
        );
    }

    let decode = Qwen35DecodeGraph::new(graph, scalars, capacity, table, token_buf, logits_buf)?;
    let mut driver = LaneDriver::Graph(Box::new(GraphLanes { decode, kv, gdn }));
    driver.reset(&client)?;
    Ok(LaneDriverBuild {
        driver,
        graph_captures,
    })
}

/// Record every lane as holding no tokens at rope position 0 and write the
/// decode scalars to match.
///
/// A claimed lane attends over one key, so the warmup and the capture both
/// record the kernels a live lane runs.
fn claim_every_lane(
    client: &CudaClient,
    scalars: &mut DeviceScalars,
    table: &mut LaneTable<CudaRuntime>,
    capacity: usize,
) -> Result<()> {
    for lane in 0..table.batch() {
        table.claim(client, lane, lane as u64, 0, 0)?;
    }
    scalars.update_lanes(client, table.write_positions(), capacity)
}

/// The stable buffers and the caches one capture records against.
struct CaptureBuffers<'a> {
    token_buf: &'a Tensor<CudaRuntime>,
    logits_buf: &'a Tensor<CudaRuntime>,
    scalars: &'a DeviceScalars,
    table: &'a LaneTable<CudaRuntime>,
    kv: &'a LayeredKvCache<CudaRuntime>,
    gdn: &'a LayeredGdnState<CudaRuntime>,
}

/// Capture one multi-lane decode step into an arena of `arena_bytes`.
///
/// The capture ends at the logits copy into `logits_buf` and picks no token:
/// the caller samples each lane's row on the host and writes the results back
/// for the next replay.
fn capture_step(
    model: &Qwen35Model<CudaRuntime>,
    client: &CudaClient,
    bufs: &CaptureBuffers<'_>,
    arena_bytes: usize,
) -> Result<CapturedGraph<CudaRuntime>> {
    CudaRuntime::capture_graph_into_with_arena(
        client,
        &[bufs.token_buf],
        &[bufs.logits_buf],
        arena_bytes,
        |c| {
            let logits = model
                .forward_qwen35_graph_mode(
                    c,
                    bufs.token_buf,
                    bufs.kv,
                    bufs.gdn,
                    bufs.scalars,
                    bufs.table,
                )
                .map_err(|e| numr::error::Error::Backend(format!("capture forward failed: {e}")))?;
            copy_into_stable(c, &logits, bufs.logits_buf)
        },
    )
    .map_err(|e| Error::InferenceError {
        reason: format!(
            "CUDA graph capture of a {}-lane decode step (arena {arena_bytes} bytes) failed: {e}; \
             retry with a larger arena when the failure names the arena",
            bufs.table.batch()
        ),
    })
}

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
//!
//! When the first guess is instead too small, capture fails with a typed
//! arena-exhaustion error naming the request and the bytes already used.
//! Capture only records without executing, so a failed attempt mutates
//! nothing and the same caches serve a retry unchanged: the first capture is
//! retried, up to a small attempt cap, against an arena sized from that
//! error's own numbers.

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
    /// Captures the build ran: 0 on the eager arm, `>= 1` on the graph arm.
    /// More than one when the first capture's arena was too small and was
    /// retried, when the loose-arena tightening pass ran, or both.
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
    build_graph_with_first_arena(spec, None)
}

/// Test-only entry point: build the graph arm with an explicit first-capture
/// arena size instead of [`first_arena_bytes`], so a test can force the
/// first capture too small and confirm the retry in [`build_graph`]
/// recovers.
#[doc(hidden)]
pub fn build_lane_driver_with_first_arena_for_test(
    spec: LaneDriverSpec<'_>,
    first_arena_bytes_override: usize,
) -> Result<LaneDriverBuild> {
    build_graph_with_first_arena(spec, Some(first_arena_bytes_override))
}

fn build_graph_with_first_arena(
    spec: LaneDriverSpec<'_>,
    first_arena_override: Option<usize>,
) -> Result<LaneDriverBuild> {
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
    let first_arena = match first_arena_override {
        Some(bytes) => bytes,
        None => first_arena_bytes(device)?,
    };
    let (mut graph, captured_arena, mut graph_captures) =
        capture_first_step(model, &client, &bufs, device, first_arena, lanes)?;
    let used = graph.arena_bytes_used().unwrap_or(captured_arena);
    tracing::info!(
        "{}-lane CUDA graph captured (arena={} used={} attempts={})",
        lanes,
        captured_arena,
        used,
        graph_captures
    );

    // Capture records without executing, so the caches still hold the same
    // state and serve a second capture unchanged.
    if arena_is_loose(used, captured_arena) {
        let tight_arena = tight_arena_bytes(used);
        drop(graph);
        graph = capture_step(model, &client, &bufs, tight_arena)
            .map_err(|e| wrap_capture_error(e, tight_arena, lanes))?;
        graph_captures += 1;
        tracing::info!(
            "{}-lane CUDA graph recaptured (arena={} used={:?})",
            lanes,
            tight_arena,
            graph.arena_bytes_used()
        );
    }

    let decode = Qwen35DecodeGraph::new(graph, scalars, capacity, table, token_buf, logits_buf)?;
    // The capture above baked every layer's device pointers into the graph,
    // so nothing may grow this cache again: freeze it before it reaches the
    // driver.
    let kv = kv.freeze();
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
///
/// Returns numr's own error unwrapped, not boostr's, so a caller can react
/// to the typed arena-exhaustion variant before deciding how to wrap it
/// (see [`capture_first_step`] and [`wrap_capture_error`]).
fn capture_step(
    model: &Qwen35Model<CudaRuntime>,
    client: &CudaClient,
    bufs: &CaptureBuffers<'_>,
    arena_bytes: usize,
) -> numr::error::Result<CapturedGraph<CudaRuntime>> {
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
                .map_err(|e| match e {
                    // A numr error that carries arena exhaustion passes through
                    // unchanged so the retry loop can still find it by type.
                    Error::Numr(inner) if inner.as_arena_exhausted().is_some() => inner,
                    other => {
                        numr::error::Error::Backend(format!("capture forward failed: {other}"))
                    }
                })?;
            copy_into_stable(c, &logits, bufs.logits_buf)
        },
    )
}

/// Wrap a capture failure that is not (or is no longer) worth retrying,
/// naming the lane count and the arena it was tried against.
fn wrap_capture_error(e: numr::error::Error, arena_bytes: usize, lanes: usize) -> Error {
    Error::InferenceError {
        reason: format!(
            "CUDA graph capture of a {lanes}-lane decode step (arena {arena_bytes} bytes) \
             failed: {e}"
        ),
    }
}

/// Attempts allowed for the first capture before giving up and reporting
/// every arena size the build tried.
const MAX_FIRST_CAPTURE_ATTEMPTS: usize = 3;

/// Capture the first graph, retrying with a larger arena when the arena
/// itself was the reason capture failed.
///
/// Reacts only to the typed [`numr::error::Error::ArenaExhausted`] (walked
/// through one layer of `AllocFailed` wrapping by
/// [`numr::error::Error::as_arena_exhausted`]) — never to message text. Any
/// other capture failure stops immediately. A capture records without
/// executing, so a failed attempt mutates none of `bufs`'s caches and the
/// same state serves the next attempt unchanged.
///
/// Each retry sizes its arena from the failed request's own numbers
/// (`used + requested`, padded the way [`tight_arena_bytes`] pads a loose
/// arena) and re-checks free device memory before committing to it, so a
/// device that cannot fit the capture fails fast instead of allocating
/// toward an OOM. The loop is capped at [`MAX_FIRST_CAPTURE_ATTEMPTS`]; the
/// final failure names every arena size tried.
///
/// Returns the graph, the arena size that captured it, and the number of
/// attempts taken (1 when the first guess already worked).
fn capture_first_step(
    model: &Qwen35Model<CudaRuntime>,
    client: &CudaClient,
    bufs: &CaptureBuffers<'_>,
    device: &CudaDevice,
    first_arena: usize,
    lanes: usize,
) -> Result<(CapturedGraph<CudaRuntime>, usize, usize)> {
    let mut arena_bytes = first_arena;
    let mut tried = Vec::with_capacity(MAX_FIRST_CAPTURE_ATTEMPTS);
    let mut attempt = 0usize;
    loop {
        attempt += 1;
        tried.push(arena_bytes);
        match capture_step(model, client, bufs, arena_bytes) {
            Ok(graph) => return Ok((graph, arena_bytes, attempt)),
            Err(e) => {
                let Some((requested, used, capacity)) = e.as_arena_exhausted() else {
                    return Err(wrap_capture_error(e, arena_bytes, lanes));
                };
                let (free, _total) = device.memory_info().map_err(|qe| Error::InferenceError {
                    reason: format!(
                        "querying device memory to retry a {lanes}-lane decode capture failed: \
                         {qe}; check that the device the graph captures on is still available"
                    ),
                })?;
                if attempt >= MAX_FIRST_CAPTURE_ATTEMPTS {
                    return Err(Error::InferenceError {
                        reason: format!(
                            "CUDA graph capture of a {lanes}-lane decode step exhausted its \
                             arena on every attempt (arenas tried in bytes: {tried:?}); the \
                             last try had a {capacity}-byte arena with {used} bytes already \
                             used and {requested} more requested, with {free} bytes free on the \
                             device; run with fewer lanes or free device memory"
                        ),
                    });
                }
                let needed = tight_arena_bytes(used + requested);
                if needed as u64 > free {
                    return Err(Error::InferenceError {
                        reason: format!(
                            "CUDA graph capture of a {lanes}-lane decode step needs a \
                             {needed}-byte arena but only {free} bytes are free on the device; \
                             run with fewer lanes or free device memory before retrying"
                        ),
                    });
                }
                arena_bytes = needed;
            }
        }
    }
}

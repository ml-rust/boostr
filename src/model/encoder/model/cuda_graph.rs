//! CUDA graph capture and replay for the encoder forward pass.
//!
//! Entry point: `try_graph_embed` — called from `Encoder::embed_inference` when
//! `feature = "cuda"` is active. Returns `Some(result)` if the runtime is CUDA
//! (routes through the graph capture cache) or `None` for all other runtimes
//! (caller falls through to the standard forward pass).
//!
//! Cache key: `(batch_size, seq_len)`. Bound: 16 entries, LRU eviction.
//! - Miss: run full encode + pool inside `CudaRuntime::capture_graph_into`, store
//!   `CapturedGraph` (holds graph + I/O tensor Arc clones) in `CapturedForward`.
//! - Hit: H2D-copy fresh inputs into captured buffers, replay graph via `cuGraphLaunch`.

use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Size of the per-graph scratch arena for graph-internal intermediate tensors.
///
/// All allocations made inside the CUDA graph capture closure are redirected
/// into this pre-allocated device buffer.  Because the buffer is allocated
/// BEFORE `cuStreamBeginCapture`, its device address is stable across graph
/// replays — this prevents `CUDA_ERROR_ILLEGAL_ADDRESS` on the second and
/// subsequent calls to `cuGraphLaunch`.
///
/// 256 MiB is a conservative bound for a 24-layer encoder at the largest
/// expected bench shape (B=64, S=512, H=1024).  Peak live intermediates at
/// any one time are roughly one-layer's worth; the bump-pointer arena LIFO
/// reclaims them between layers, so the peak working set is well under 64 MiB.
/// The 256 MiB ceiling gives comfortable headroom for larger models and
/// longer sequences.
///
/// If this triggers OOM for a truly large model, reduce the batch size or
/// increase this constant.
const ENCODER_ARENA_BYTES: usize = 256 * 1024 * 1024;

use crate::error::{Error, Result};
use crate::model::encoder::config::EncoderConfig;
use crate::model::encoder::model::graph_cache::CapturedForward;
use crate::model::encoder::model::{Encoder, EncoderClient, Pooling};

/// Called from `Encoder::embed_inference` for every runtime when `cuda` feature is on.
///
/// Returns `Some(Result<Tensor<R>>)` if the runtime is CUDA and we handled the call
/// through the graph cache. Returns `None` for non-CUDA runtimes; caller uses
/// the standard forward path.
pub fn try_graph_embed<R, C>(
    encoder: &Encoder<R>,
    client: &C,
    input_ids: &Tensor<R>,
    attention_mask: Option<&Tensor<R>>,
) -> Option<Result<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: EncoderClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    // Only activate for CudaRuntime.
    if R::name() != "cuda" {
        return None;
    }

    // SAFETY: We confirmed R::name() == "cuda". `CudaRuntime::name()` returns "cuda"
    // and is the only runtime with that name. Therefore R == CudaRuntime, which means:
    //   - Encoder<R>    has identical layout to Encoder<CudaRuntime>
    //   - Tensor<R>     has identical layout to Tensor<CudaRuntime>
    //   - C             is CudaClient (the only client for CudaRuntime)
    //
    // The pointer casts below are safe because monomorphic generic types with the
    // same concrete type parameter have identical memory layouts.
    let enc: &Encoder<CudaRuntime> =
        unsafe { &*(encoder as *const Encoder<R> as *const Encoder<CudaRuntime>) };
    let ids: &Tensor<CudaRuntime> =
        unsafe { &*(input_ids as *const Tensor<R> as *const Tensor<CudaRuntime>) };
    let mask: Option<&Tensor<CudaRuntime>> =
        attention_mask.map(|m| unsafe { &*(m as *const Tensor<R> as *const Tensor<CudaRuntime>) });
    let cc: &CudaClient = unsafe { &*(client as *const C as *const CudaClient) };

    let result = embed_cached(enc, cc, ids, mask);

    Some(result.map(|t| {
        // Transmute Tensor<CudaRuntime> back to Tensor<R>. Safe for same reason.
        unsafe { std::mem::transmute::<Tensor<CudaRuntime>, Tensor<R>>(t) }
    }))
}

// ---------------------------------------------------------------------------
// Cache dispatch
// ---------------------------------------------------------------------------

fn embed_cached(
    enc: &Encoder<CudaRuntime>,
    client: &CudaClient,
    input_ids: &Tensor<CudaRuntime>,
    attention_mask: Option<&Tensor<CudaRuntime>>,
) -> Result<Tensor<CudaRuntime>> {
    let shape = input_ids.shape().to_vec();
    let batch = if shape.len() == 2 { shape[0] } else { 1 };
    let seq_len = *shape.last().ok_or_else(|| Error::ModelError {
        reason: "input_ids must have at least 1 dimension".into(),
    })?;

    // Compute position IDs on the host BEFORE capture — the XLM-RoBERTa path
    // reads input_ids back from the device (D2H), which cannot happen inside
    // a stream-capture region.
    let flat_ids: Vec<i64> = input_ids.to_vec();
    let pos_flat = enc.compute_position_ids_host(&flat_ids, batch, seq_len);

    let cache = &enc.forward_cache;

    if cache.contains(batch, seq_len) {
        let result = cache
            .with_entry(batch, seq_len, |entry| {
                replay(entry, &flat_ids, &pos_flat, attention_mask, batch, seq_len)
            })
            .ok_or_else(|| Error::ModelError {
                reason: "CUDA graph cache entry evicted between contains() and replay".into(),
            })?;
        return result;
    }

    capture_and_run(
        enc,
        client,
        input_ids,
        attention_mask,
        &flat_ids,
        &pos_flat,
        batch,
        seq_len,
    )
}

// ---------------------------------------------------------------------------
// Capture path
// ---------------------------------------------------------------------------

fn capture_and_run(
    enc: &Encoder<CudaRuntime>,
    client: &CudaClient,
    input_ids: &Tensor<CudaRuntime>,
    attention_mask: Option<&Tensor<CudaRuntime>>,
    flat_ids: &[i64],
    pos_flat: &[i64],
    batch: usize,
    seq_len: usize,
) -> Result<Tensor<CudaRuntime>> {
    let device = input_ids.device();
    let hidden_size = enc.config.hidden_size;

    let pos_shape: Vec<usize> = if input_ids.shape().len() == 2 {
        vec![batch, seq_len]
    } else {
        vec![seq_len]
    };

    // Allocate stable-address I/O buffers BEFORE capture begins.
    // The graph encodes these device pointers; they must not move.
    let input_ids_buf = Tensor::<CudaRuntime>::from_slice(flat_ids, &[batch, seq_len], device);
    let pos_ids_buf = Tensor::<CudaRuntime>::from_slice(pos_flat, &pos_shape, device);

    let flat_mask: Vec<f32> = attention_mask
        .map(|m| m.to_vec())
        .unwrap_or_else(|| vec![1.0f32; batch * seq_len]);
    let mask_buf = Tensor::<CudaRuntime>::from_slice(&flat_mask, &[batch, seq_len], device);

    // Stable output buffer [B, hidden] — allocated OUTSIDE capture so it is
    // NOT subject to AUTO_FREE_ON_LAUNCH. The graph writes into it via D2D copy.
    let stable_out = Tensor::<CudaRuntime>::from_slice(
        &vec![0.0f32; batch * hidden_size],
        &[batch, hidden_size],
        device,
    );
    let stable_out_ptr = stable_out.ptr();

    // Pre-allocate a `[1]` f32 tensor holding 1.0 OUTSIDE capture.
    //
    // This prevents `Tensor::from_slice(&[1.0f32], ...)` from being called
    // inside the graph capture closure.  An inline `from_slice` inside the
    // closure would record an H2D memcpy node with the stack-temporary's
    // address as the source.  On replay that stack frame is gone →
    // CUDA_ERROR_ILLEGAL_ADDRESS.  Allocating the scalar BEFORE capture and
    // passing it into the closure gives the graph a stable device address.
    let ones_scalar = Tensor::<CudaRuntime>::from_slice(&[1.0f32], &[1], device);

    let ids_ref = &input_ids_buf;
    let pos_ref = &pos_ids_buf;
    let mask_ref = &mask_buf;
    let ones_ref = &ones_scalar;

    // Capture: encode_inference_with_pos → pool → D2D copy into stable_out.
    //
    // inputs  = [input_ids_buf, pos_ids_buf, mask_buf]  (fixed-address read buffers)
    // outputs = [stable_out]                            (fixed-address write buffer)
    //
    // The closure writes into `stable_out` (outside the capture region) via an
    // in-graph D2D copy, so it is NOT subject to AUTO_FREE_ON_LAUNCH.
    //
    // Arena: all intermediate tensors allocated INSIDE the closure (the ~22
    // intermediates per encoder forward) are redirected into a pre-allocated
    // device buffer.  Because the arena buffer was allocated BEFORE capture
    // begins, its address is stable across replays — no CUDA_ERROR_ILLEGAL_ADDRESS.
    let captured = CudaRuntime::capture_graph_into_with_arena(
        client,
        &[&input_ids_buf, &pos_ids_buf, &mask_buf],
        &[&stable_out],
        ENCODER_ARENA_BYTES,
        |cc| {
            let hidden = enc
                .encode_inference_with_pos(cc, ids_ref, pos_ref, Some(mask_ref))
                .map_err(|e| numr::error::Error::Backend(format!("encoder forward: {e:#}")))?;

            let pooled = pool_hidden(
                cc,
                &hidden,
                Some(mask_ref),
                &enc.pooling,
                &enc.config,
                ones_ref,
            )
            .map_err(|e| numr::error::Error::Backend(format!("pooling: {e:#}")))?;

            let n_bytes = batch * hidden_size * std::mem::size_of::<f32>();
            CudaRuntime::copy_within_device(pooled.ptr(), stable_out_ptr, n_bytes, device)?;

            Ok(())
        },
    )?;

    // Insert before reading — cache takes ownership of the CapturedGraph.
    enc.forward_cache
        .insert(batch, seq_len, CapturedForward::new(captured));

    // Execute the captured graph so stable_out contains real data, then return
    // a clone of the output buffer. The graph was only *recorded* above, not run.
    enc.forward_cache
        .with_entry(batch, seq_len, |e| {
            e.launch().map_err(Error::Numr)?;
            Ok(e.output_buf().clone())
        })
        .ok_or_else(|| Error::ModelError {
            reason: "CUDA graph cache entry missing immediately after insert".into(),
        })?
}

// ---------------------------------------------------------------------------
// Replay path
// ---------------------------------------------------------------------------

fn replay(
    entry: &CapturedForward,
    flat_ids: &[i64],
    pos_flat: &[i64],
    attention_mask: Option<&Tensor<CudaRuntime>>,
    batch: usize,
    seq_len: usize,
) -> Result<Tensor<CudaRuntime>> {
    let device = entry.input_ids_buf().device();

    // H2D: overwrite token id buffer (stream-ordered, records before graph launch).
    CudaRuntime::copy_to_device(cast_i64(flat_ids), entry.input_ids_buf().ptr(), device)
        .map_err(Error::Numr)?;

    // H2D: overwrite position id buffer.
    CudaRuntime::copy_to_device(cast_i64(pos_flat), entry.pos_ids_buf().ptr(), device)
        .map_err(Error::Numr)?;

    // H2D: overwrite attention mask buffer.
    let flat_mask: Vec<f32> = attention_mask
        .map(|m| m.to_vec())
        .unwrap_or_else(|| vec![1.0f32; batch * seq_len]);
    CudaRuntime::copy_to_device(cast_f32(&flat_mask), entry.mask_buf().ptr(), device)
        .map_err(Error::Numr)?;

    // Single graph launch — replaces ~192 individual kernel dispatches.
    entry.launch().map_err(Error::Numr)?;

    // stable_out was written by the D2D copy node inside the graph.
    Ok(entry.output_buf().clone())
}

// ---------------------------------------------------------------------------
// Pooling (runs inside graph capture for clean graph membership)
// ---------------------------------------------------------------------------

/// Pooling helper for use inside a CUDA graph capture closure.
///
/// `ones_scalar`: a pre-allocated `[1]` f32 tensor containing the value `1.0`
/// that was allocated OUTSIDE the capture region.  This prevents
/// `Tensor::from_slice(&[1.0f32], ...)` from being called inside the closure,
/// which would record an H2D memcpy node with a stale stack address into the
/// graph — causing `CUDA_ERROR_ILLEGAL_ADDRESS` on replay.
fn pool_hidden(
    client: &CudaClient,
    hidden: &Tensor<CudaRuntime>,
    mask: Option<&Tensor<CudaRuntime>>,
    pooling: &Pooling,
    config: &EncoderConfig,
    ones_scalar: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let hidden_var = Var::new(hidden.clone(), false);

    match pooling {
        Pooling::Mean => {
            if let Some(m) = mask {
                let mask_shape = m.shape().to_vec();
                let b = mask_shape[0];
                let s = mask_shape[1];

                let mask_3d = m.reshape(&[b, s, 1]).map_err(Error::Numr)?;
                let masked =
                    BinaryOps::mul(client, hidden_var.tensor(), &mask_3d).map_err(Error::Numr)?;
                let summed = ReduceOps::sum(client, &masked, &[1], false).map_err(Error::Numr)?;
                let counts = ReduceOps::sum(client, m, &[1], true).map_err(Error::Numr)?;
                // Use the pre-allocated `ones_scalar` (allocated OUTSIDE capture)
                // instead of `Tensor::from_slice(&[1.0f32], ...)` here.
                // An inline from_slice would bake a host stack pointer into the
                // graph's H2D memcpy node, causing CUDA_ERROR_ILLEGAL_ADDRESS on replay.
                let counts =
                    BinaryOps::maximum(client, &counts, ones_scalar).map_err(Error::Numr)?;
                let _ = config.hidden_size;
                Ok(BinaryOps::div(client, &summed, &counts).map_err(Error::Numr)?)
            } else {
                Ok(ReduceOps::mean(client, hidden_var.tensor(), &[1], false)
                    .map_err(Error::Numr)?)
            }
        }
        Pooling::Cls => {
            let cls = var_narrow(&hidden_var, 1, 0, 1).map_err(Error::Numr)?;
            let cls = Var::new(cls.tensor().contiguous(), false);
            let sh = cls.shape().to_vec();
            let b = sh[0];
            let h = sh[2];
            Ok(var_reshape(&cls, &[b, h])
                .map_err(Error::Numr)?
                .tensor()
                .clone())
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-cast helpers
// ---------------------------------------------------------------------------

#[inline]
fn cast_i64(data: &[i64]) -> &[u8] {
    // SAFETY: i64 is Pod; no padding; pointer is valid for lifetime of slice.
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<i64>(),
        )
    }
}

#[inline]
fn cast_f32(data: &[f32]) -> &[u8] {
    // SAFETY: f32 is Pod; no padding; pointer is valid for lifetime of slice.
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

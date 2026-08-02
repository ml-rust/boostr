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

use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::model::encoder::model::graph_cache::CapturedForward;
use crate::model::encoder::model::layer::SpanMasks;
use crate::model::encoder::model::pooling::pool_padded;
use crate::model::encoder::model::{Encoder, EncoderClient};

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

#[allow(clippy::too_many_arguments)]
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

    // Sliding-window / causal span masks, built here for the same reason: the
    // host-to-device copy must not happen inside the capture region.
    let span_masks = SpanMasks::<CudaRuntime>::build(&enc.config, seq_len, device);

    let ids_ref = &input_ids_buf;
    let pos_ref = &pos_ids_buf;
    let mask_ref = &mask_buf;
    let ones_ref = &ones_scalar;
    let span_ref = &span_masks;

    // Every device buffer the graph reads must be handed to `capture_graph_into`
    // as an input: it clones them into the `CapturedGraph`, which is what keeps
    // the allocations alive for the graph's lifetime. A buffer left out here is
    // freed when this function returns, and replay then reads freed device
    // memory. `graph_cache` only ever indexes inputs[0..=2], so the retained
    // extras can be appended.
    let mut capture_inputs: Vec<&Tensor<CudaRuntime>> =
        vec![&input_ids_buf, &pos_ids_buf, &mask_buf, &ones_scalar];
    capture_inputs.extend(span_masks.tensors());

    // Capture: encode_inference_with_pos → pool → D2D copy into stable_out.
    //
    // inputs  = [input_ids_buf, pos_ids_buf, mask_buf]  (fixed-address read buffers)
    // outputs = [stable_out]                            (fixed-address write buffer)
    //
    // The closure writes into `stable_out` (outside the capture region) via an
    // in-graph D2D copy, so it is NOT subject to AUTO_FREE_ON_LAUNCH.
    //
    // Intermediates: all tensors allocated INSIDE the closure (the ~22
    // intermediates per encoder forward) become driver-managed MEM_ALLOC graph
    // nodes via cuMemAllocAsync. AUTO_FREE_ON_LAUNCH reclaims them at the
    // end of each replay, so they are correctly managed across launches
    // without a pre-allocated arena.
    let captured =
        CudaRuntime::capture_graph_into(client, &capture_inputs, &[&stable_out], |cc| {
            let hidden = enc
                .encode_inference_with_pos(cc, ids_ref, pos_ref, Some(mask_ref), span_ref)
                .map_err(|e| numr::error::Error::Backend(format!("encoder forward: {e:#}")))?;

            let pooled = pool_padded(cc, &hidden, Some(mask_ref), enc.pooling, Some(ones_ref))
                .map_err(|e| numr::error::Error::Backend(format!("pooling: {e:#}")))?;

            let n_bytes = batch * hidden_size * std::mem::size_of::<f32>();
            CudaRuntime::copy_within_device(pooled.ptr(), stable_out_ptr, n_bytes, device)?;

            Ok(())
        })?;

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
// Host byte views for H2D copies
// ---------------------------------------------------------------------------
//
// `Runtime::copy_to_device` takes the source as `&[u8]` and copies it verbatim;
// it performs no element conversion. The captured buffers were built with
// `Tensor::from_slice` over `&[i64]` (ids/positions) and `&[f32]` (mask), so
// replay must hand it the raw little-endian bytes of those same element types.
// `bytemuck` is not a dependency under the `cuda` feature, so the byte views are
// taken directly; both types are plain-old-data with no padding, and `u8` has
// alignment 1, so the reinterpretation is sound.

fn cast_i64(data: &[i64]) -> &[u8] {
    // SAFETY: `i64` is POD with no padding; the view covers exactly the same
    // bytes as `data` and borrows it for the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn cast_f32(data: &[f32]) -> &[u8] {
    // SAFETY: `f32` is POD with no padding; the view covers exactly the same
    // bytes as `data` and borrows it for the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
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

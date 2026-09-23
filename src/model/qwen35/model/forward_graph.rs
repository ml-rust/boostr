//! [`Qwen35Model`] CUDA graph-mode decode forward.
//!
//! One captured graph replays one token. Everything that changes between
//! replays lives at an address allocated before capture:
//!
//! | Input                     | Written how                                    |
//! |---------------------------|------------------------------------------------|
//! | `input_ids` `[B, 1]` i64  | caller, D2D async from the previous output      |
//! | `DeviceScalars`           | caller, `DeviceScalars::update_lanes(lens, cap)`|
//! | `kv_start`/`kv_end` `[B]` | caller, `LaneTable::advance`                    |
//! | IMROPE positions `[4,B,1]`| caller, `LaneTable::advance`                    |
//! | KV cache k/v buffers      | graph, `kv_insert` at `write_pos[b]`            |
//! | GDN conv/ssm buffers      | graph, `GdnState::update_shared`                |
//!
//! The layer walk keeps the same `attn_idx` / `gdn_idx` counters as
//! `forward_layers`.

#[cfg(feature = "cuda")]
use super::build::{Qwen35Block, Qwen35Model};
#[cfg(feature = "cuda")]
use super::forward::ffn_residual;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::decode_graph::DeviceScalars;
#[cfg(feature = "cuda")]
use crate::inference::lanes::LaneTable;
#[cfg(feature = "cuda")]
use crate::inference::{LayeredGdnState, LayeredKvCache};
#[cfg(feature = "cuda")]
use crate::model::hybrid::Qwen35GraphStep;

#[cfg(feature = "cuda")]
impl Qwen35Model<numr::runtime::cuda::CudaRuntime> {
    /// Graph-mode decode forward: logits `[batch, 1, vocab_size]` whose
    /// address is graph-managed. Read them through a captured copy into a
    /// stable buffer (`copy_into_stable`).
    ///
    /// One lane per sequence. Lane `b` holds its own tokens from cache
    /// position 0 with its own end, so a replay serves lanes of different
    /// live lengths and a free lane reads no key.
    ///
    /// The caller must:
    ///
    /// 1. Pre-allocate `kv_cache` at full capacity, one lane per sequence,
    ///    and prefill every occupied lane eagerly.
    /// 2. Prefill `gdn_state` eagerly with the same tokens, lane for lane.
    /// 3. Write `device_scalars` for each lane's KV slot and `lanes` for
    ///    each lane's key range and rope position before every replay. A
    ///    lane's slot and its rope position agree for a text-only context
    ///    and drift apart after an image.
    /// 4. Advance the lanes outside the graph: this forward does not read
    ///    the CPU-side cache length, and does not advance it.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `kv_cache` or `gdn_state` hold fewer layers
    /// than the model has of that kind; block errors propagate.
    pub fn forward_qwen35_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &LayeredKvCache<numr::runtime::cuda::CudaRuntime>,
        gdn_state: &LayeredGdnState<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &DeviceScalars,
        lanes: &LaneTable<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>> {
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        let mut attn_idx = 0usize;
        let mut gdn_idx = 0usize;

        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                Qwen35Block::Attention(layer) => {
                    let cache = kv_cache
                        .layer(attn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "qwen35 graph mode: KV cache missing for layer {i} (attn_idx={attn_idx})"
                            ),
                        })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = layer.mixer.forward_graph_mode(
                        client,
                        &normed,
                        &self.rope,
                        cache,
                        &Qwen35GraphStep {
                            positions: lanes.positions(),
                            device_scalars,
                            lanes: lanes.step(),
                        },
                    )?;
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    attn_idx += 1;
                }
                Qwen35Block::Gdn(layer) => {
                    let state = gdn_state.layer(gdn_idx).ok_or_else(|| Error::ModelError {
                        reason: format!(
                            "qwen35 graph mode: GDN state missing for layer {i} (gdn_idx={gdn_idx})"
                        ),
                    })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = layer.mixer.forward_graph_mode(client, &normed, state)?;
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    gdn_idx += 1;
                }
            }
        }

        let hidden = self.norm.forward(client, &hidden)?;
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
    }
}

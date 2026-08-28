//! Incremental (KV-cached) decode for VoxCPM2's MiniCPM4 `base_lm`.
//!
//! The sibling [`MiniCpm4Model::forward`] recomputes every position on every
//! call. This file adds the generation path: prime a cache from a prefix once,
//! then advance one position at a time.
//!
//! ```text
//! let mut cache = model.new_kv_cache(batch, max_length)?;   // preallocated
//! let prefix_out = model.prefill(&client, &prefix_embeds, &mut cache)?;
//! let step_out   = model.decode_step(&client, &embed, &mut cache, position)?;
//! ```
//!
//! # Cache-shape equivalence with the reference
//!
//! The reference (`voxcpm/modules/minicpm4/cache.py`) preallocates the cache to
//! `max_length` and masks with `arange(max_length) <= position_id`;
//! [`KvCache`](crate::inference::KvCache) exposes only the slots actually
//! written, and the shared causal-mask builder admits `0..=position`. Both
//! select the SAME keys, and only while positions are written in order from 0.
//! [`MiniCpm4Attention::forward_cached`] carries the full argument at the
//! masking site; [`MiniCpm4Model::decode_step`] enforces the ordering premise.
//!
//! # Bounds
//!
//! The reference's `step()` raises when `current_length >= max_length`, and
//! nothing deeper in `forward_step` re-checks — an out-of-range position
//! silently corrupts the cache write. Here [`MiniCpm4Model::decode_step`]
//! rejects `position >= max_length` with an [`Error::InvalidArgument`] before
//! any tensor is touched; there is no panicking index on this path.
//!
//! # Numerical agreement
//!
//! The step-wise and full-sequence paths run the same weights, the same
//! precomputed RoPE tables at the same absolute positions, and select the same
//! keys — but reduce over the key axis in a different order (one row at a time
//! versus a full matrix). They agree to roughly `1e-4` absolute, not bitwise.
//! The reference's own two paths differ by `9.9e-5` for the same reason.

use crate::error::{Error, Result};
use crate::inference::{LayeredKvCache, LayeredKvCacheConfig};
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::model::MiniCpm4Model;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// First layer's attention block, or a `ModelError` for a zero-layer model.
    ///
    /// The cache geometry (`num_kv_heads`, `head_dim`) and the K/V dtype are
    /// read from it rather than taken as arguments, so a caller cannot size a
    /// cache the projections will not fit.
    fn first_attention(&self) -> Result<&MiniCpm4Attention<R>> {
        self.layers
            .first()
            .map(|layer| &layer.self_attn)
            .ok_or_else(|| Error::ModelError {
                reason: "MiniCPM4 model has no layers; cannot size a KV cache".to_string(),
            })
    }

    /// Number of RoPE positions the loader precomputed
    /// (`max_position_embeddings`), or `None` for a NoPE (`no_rope`) stack,
    /// which has no table. A cache may not outrun a table that exists.
    fn rope_positions(&self) -> Option<usize> {
        self.rope.as_ref().map(|rope| rope.cos_cache().shape()[0])
    }

    /// Allocate a KV cache for this model, sized by the caller.
    ///
    /// `max_length` slots are allocated UP FRONT (initial capacity ==
    /// `max_length`), matching the reference's preallocated cache: the
    /// generation loop then never triggers a reallocation mid-decode.
    ///
    /// Dtype and device come from the first layer's `k_proj` — see
    /// [`MiniCpm4Attention::kv_dtype_device`], which reports the dtype of
    /// that projection's OUTPUT (F32 for a quantized weight) rather than
    /// reading a weight tensor that a packed weight does not have.
    ///
    /// Errors when `batch_size` or `max_length` is zero, or when `max_length`
    /// exceeds the precomputed RoPE table (a longer cache could be filled but
    /// never rotated). A NoPE (`no_rope`) stack has no table and rotates
    /// nothing, so only the zero check applies there.
    pub fn new_kv_cache(&self, batch_size: usize, max_length: usize) -> Result<LayeredKvCache<R>>
    where
        R::Client: IndexingOps<R>,
    {
        if batch_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "batch_size",
                reason: "expected at least 1, got 0".to_string(),
            });
        }
        if max_length == 0 {
            return Err(Error::InvalidArgument {
                arg: "max_length",
                reason: "expected at least 1, got 0".to_string(),
            });
        }
        if let Some(rope_positions) = self.rope_positions()
            && max_length > rope_positions
        {
            return Err(Error::InvalidArgument {
                arg: "max_length",
                reason: format!(
                    "expected 1..={rope_positions} (the precomputed RoPE table length), got {max_length}"
                ),
            });
        }

        let attn = self.first_attention()?;
        let (dtype, device) = attn.kv_dtype_device()?;
        let config = LayeredKvCacheConfig {
            batch_size,
            num_kv_heads: attn.num_kv_heads,
            initial_capacity: max_length,
            max_seq_len: max_length,
            head_dim: attn.head_dim,
            dtype,
        };
        LayeredKvCache::new(self.layers.len(), &config, device)
    }

    /// Run the full prefix and populate `kv_cache` with its K/V.
    ///
    /// `inputs_embeds: [batch, seq, hidden_size]` -> `[batch, seq,
    /// hidden_size]`, the same value and shape
    /// [`forward`](MiniCpm4Model::forward) returns for the same input.
    ///
    /// The cache is RESET first, mirroring the reference's `fill_caches`, which
    /// zeroes the buffers before copying the primed length in. Call this ONCE
    /// per sequence, then [`decode_step`](Self::decode_step) from
    /// `position == seq`.
    pub fn prefill<C>(
        &self,
        client: &C,
        inputs_embeds: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let shape = inputs_embeds.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected 3D [batch, seq, hidden_size], got {}D",
                    shape.len()
                ),
            });
        }
        self.check_cache(kv_cache, shape[0], shape[2], shape[1], 0)?;

        kv_cache.reset();
        self.forward_cached(client, inputs_embeds, kv_cache, 0)
    }

    /// Advance one position.
    ///
    /// `embed: [batch, hidden_size]` -> `[batch, hidden_size]`. The step entry
    /// point takes and returns the SQUEEZED 2D shape because that is what the
    /// reference's `forward_step` is handed (`curr_embed[:, 0, :]`); a caller
    /// porting a generation loop passes the same tensor it already has, with no
    /// unsqueeze on either side. The `[batch, 1, hidden]` view is rebuilt
    /// internally for the layer stack.
    ///
    /// `position` is the ABSOLUTE index of this embedding and must equal
    /// `kv_cache.seq_len()` — the slot the cache will write. Accepting any
    /// other index would rotate the query at one position while filing its key
    /// at another, which stays shape-valid and silently computes a different
    /// model.
    ///
    /// Errors (never panics, never writes out of range) when `position`
    /// reaches the cache's `max_length`, when it disagrees with the cache
    /// length, or when the shape or batch does not match.
    pub fn decode_step<C>(
        &self,
        client: &C,
        embed: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let shape = embed.shape().to_vec();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "embed",
                reason: format!(
                    "expected 2D [batch, hidden_size] (one position), got {}D",
                    shape.len()
                ),
            });
        }
        let (batch, hidden) = (shape[0], shape[1]);
        self.check_cache(kv_cache, batch, hidden, 1, position)?;

        // The write-order rule that makes this cache equivalent to the
        // reference's preallocated one: `position` must be the slot the cache
        // will actually write. `prefill` is exempt because it resets first.
        // `check_cache` above already required layer 0 to exist, so this
        // cannot be `None` in practice; propagate rather than default to 0,
        // which would silently mask that invariant if it were ever broken.
        let filled = kv_cache
            .layer(0)
            .ok_or_else(|| Error::ModelError {
                reason: "KV cache missing layer 0 after validation".to_string(),
            })?
            .seq_len();
        if filled != position {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "expected {filled} (the next free cache slot), got {position}; \
                     positions must be written in order from 0"
                ),
            });
        }

        let x = var_reshape(embed, &[batch, 1, hidden]).map_err(Error::Numr)?;
        let out = self.forward_cached(client, &x, kv_cache, position)?;
        var_reshape(&out, &[batch, hidden]).map_err(Error::Numr)
    }

    /// Shared validation for both cached entry points.
    ///
    /// Checks the hidden width, the cache's layer count and batch, and that
    /// `position + seq` fits inside `max_length`. The write-order rule
    /// (`position == kv_cache.seq_len()`) belongs to
    /// [`decode_step`](Self::decode_step) alone — [`prefill`](Self::prefill)
    /// resets the cache and so always starts at 0.
    fn check_cache(
        &self,
        kv_cache: &LayeredKvCache<R>,
        batch: usize,
        hidden: usize,
        seq: usize,
        position: usize,
    ) -> Result<()>
    where
        R::Client: IndexingOps<R>,
    {
        if hidden != self.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!("expected hidden_size {}, got {hidden}", self.hidden_size),
            });
        }
        if kv_cache.num_layers() != self.layers.len() {
            return Err(Error::InvalidArgument {
                arg: "kv_cache",
                reason: format!(
                    "expected a cache with {} layers, got {}",
                    self.layers.len(),
                    kv_cache.num_layers()
                ),
            });
        }
        let layer = kv_cache.layer(0).ok_or_else(|| Error::InvalidArgument {
            arg: "kv_cache",
            reason: "expected at least 1 layer, got an empty cache".to_string(),
        })?;
        if layer.batch_size() != batch {
            return Err(Error::InvalidArgument {
                arg: "kv_cache",
                reason: format!(
                    "expected a cache with batch {batch}, got {}",
                    layer.batch_size()
                ),
            });
        }
        // The reference's `step()` raises at `current_length >= max_length`;
        // this is the same guard, applied before any write.
        let max_length = layer.max_seq_len();
        if position + seq > max_length {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "position {position} plus {seq} new position(s) exceeds the cache max_length {max_length}"
                ),
            });
        }
        Ok(())
    }

    /// The cached layer stack: `[batch, seq, hidden]` covering absolute
    /// positions `position..position + seq` -> `[batch, seq, hidden]` after the
    /// final `norm`.
    ///
    /// Same layer order and same final norm as
    /// [`forward`](MiniCpm4Model::forward); only attention differs. Prefill
    /// (`seq == prefix`, `position == 0`) and a decode step (`seq == 1`) both
    /// run through here, so the two cached shapes cannot drift apart.
    fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            h = layer.forward_cached(client, &h, self.rope.as_ref(), cache, position)?;
        }
        self.norm.forward(client, &h)
    }
}

#[cfg(test)]
mod tests;

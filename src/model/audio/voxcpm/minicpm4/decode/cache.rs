//! KV cache sizing for [`MiniCpm4Model`] and the validation both cached entry
//! points share.

use crate::error::{Error, Result};
use crate::inference::{LayeredKvCache, LayeredKvCacheConfig};
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::model::MiniCpm4Model;
use numr::dtype::DType;
use numr::ops::IndexingOps;
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

    /// Shared validation for both cached entry points.
    ///
    /// Checks the hidden width, the cache's layer count and batch, and that
    /// `position + seq` fits inside `max_length`. The write-order rule
    /// (`position == kv_cache.seq_len()`) belongs to
    /// [`decode_step`](Self::decode_step) alone — [`prefill`](Self::prefill)
    /// resets the cache and so always starts at 0.
    pub(super) fn check_cache(
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
}

#[cfg(test)]
mod tests {
    use crate::model::audio::voxcpm::minicpm4::model::tests::{tiny_model, tiny_nope_model};
    use crate::test_utils::cpu_setup;

    #[test]
    fn new_kv_cache_rejects_degenerate_sizes() {
        let (_client, device) = cpu_setup();
        let model = tiny_model(&device);
        assert!(model.new_kv_cache(0, 4).is_err(), "zero batch accepted");
        assert!(
            model.new_kv_cache(1, 0).is_err(),
            "zero max_length accepted"
        );
        // The tiny model's RoPE table is 16 positions long.
        assert!(
            model.new_kv_cache(1, 17).is_err(),
            "max_length beyond the RoPE table accepted"
        );
        assert!(model.new_kv_cache(1, 16).is_ok());
    }

    /// A NoPE stack owns no RoPE table, so the cache length it can serve is
    /// bounded by the cache alone.
    #[test]
    fn nope_kv_cache_is_not_bounded_by_a_rope_table() {
        let (_client, device) = cpu_setup();
        let model = tiny_nope_model(&device);
        // 17 exceeds the 16-position table the rotary tiny model carries; this one
        // has no table to exceed.
        assert!(model.new_kv_cache(1, 17).is_ok());
        assert!(
            model.new_kv_cache(1, 0).is_err(),
            "zero max_length accepted"
        );
    }
}

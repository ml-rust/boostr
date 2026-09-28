//! Full Whisper decoder: embeddings → N layers → final LN → tied logits head.

use super::cache::{DecoderCache, DecoderLayerCache};
use super::capture::CrossAttentionCapture;
use super::helpers::load_layernorm;
use super::layer::WhisperDecoderLayer;
use crate::error::{Error, Result};
use crate::model::config::AudioConfig;
use crate::nn::{Embedding, LayerNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Whisper decoder: token+positional embeddings → N layers with cross-attn → final LN → tied `proj_out`.
pub struct WhisperDecoder<R: Runtime> {
    embed_tokens: Embedding<R>,
    /// Shape `[max_target_positions, hidden]`, added to token embeddings at the
    /// current decoding position(s).
    embed_positions: Tensor<R>,
    layers: Vec<WhisperDecoderLayer<R>>,
    layer_norm: LayerNorm<R>,
    /// Tied to `embed_tokens.weight` (shape `[vocab, hidden]`). We keep an owned
    /// handle here and use matmul with transposed embedding matrix as the logits head.
    tied_out_weight: Tensor<R>,
    vocab_size: usize,
    hidden_size: usize,
}

impl<R: Runtime<DType = DType>> WhisperDecoder<R> {
    /// Load a Whisper decoder from a VarBuilder rooted at the decoder's prefix
    /// (typically `vb.pp("model").pp("decoder")` for HF checkpoints).
    ///
    /// `proj_out.weight` is assumed tied to `embed_tokens.weight`.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &AudioConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_heads;
        let num_layers = config.decoder_layer_count();

        let mut tok_vb = vb.pp("embed_tokens");
        let tok_weight = tok_vb.take_tensor("weight")?;
        drop(tok_vb);
        let tied_out_weight = tok_weight.clone();
        let embed_tokens = Embedding::new(tok_weight, false);

        let mut pos_vb = vb.pp("embed_positions");
        let embed_positions = pos_vb.take_tensor("weight")?;
        drop(pos_vb);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let mut layer_vb = vb.pp(&format!("layers.{i}"));
            layers.push(WhisperDecoderLayer::from_varbuilder(
                &mut layer_vb,
                hidden,
                num_heads,
            )?);
        }

        let layer_norm = load_layernorm(vb, "layer_norm")?;

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            tied_out_weight,
            vocab_size: config.vocab_size,
            hidden_size: hidden,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of decoder layers (for allocating a matching `DecoderCache`).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Allocate an empty KV cache sized for this decoder.
    pub fn new_cache(&self) -> DecoderCache<R> {
        DecoderCache {
            layers: (0..self.layers.len())
                .map(|_| DecoderLayerCache::default())
                .collect(),
        }
    }

    /// Forward pass using a KV cache. Suitable for both prefill (multi-token
    /// `input_ids`) and incremental decoding (single-token `input_ids`).
    ///
    /// Returns logits for the **current** input window only: `[B, T, vocab]`.
    ///
    /// Delegates to [`Self::forward_with_cache_capturing`] with no capture
    /// sink, so this path pays no extra cost for the capture feature.
    pub fn forward_with_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
        cache: &mut DecoderCache<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        self.forward_with_cache_capturing(
            client,
            input_ids,
            encoder_out,
            position_offset,
            cache,
            None,
        )
    }

    /// Same as [`Self::forward_with_cache`], with an optional
    /// [`CrossAttentionCapture`] that collects each requested layer's
    /// cross-attention probabilities for this call.
    pub(crate) fn forward_with_cache_capturing<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
        cache: &mut DecoderCache<R>,
        mut capture: Option<&mut CrossAttentionCapture<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "decoder cache layer count mismatch"
        );

        let shape = input_ids.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let tok = self.embed_tokens.forward(client, input_ids)?;

        let pos = self
            .embed_positions
            .narrow(0, position_offset, seq_len)
            .map_err(Error::Numr)?
            .reshape(&[1, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .broadcast_to(&[batch, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .contiguous()?;

        let mut hidden = client.add(tok.tensor(), &pos).map_err(Error::Numr)?;

        for (i, (layer, layer_cache)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            let sink = match capture.as_mut() {
                Some(cap) => cap.slots.get_mut(i).and_then(|slot| slot.as_mut()),
                None => None,
            };
            hidden = layer.forward_with_cache_capturing(
                client,
                &hidden,
                encoder_out,
                layer_cache,
                sink,
            )?;
        }

        let normed = self.layer_norm.forward(client, &Var::new(hidden, false))?;
        let hidden = normed.tensor().clone();

        // w_t stays a view so the matmul reads the weight transposed without a copy.
        let w_t = self.tied_out_weight.transpose(0, 1).map_err(Error::Numr)?;
        let logits = client.matmul(&hidden, &w_t).map_err(Error::Numr)?;
        Ok(logits)
    }

    /// Forward pass over a full decoder sequence, producing logits `[B, T, vocab]`.
    ///
    /// - `input_ids`: `[B, T]` token ids
    /// - `encoder_out`: `[B, S, hidden]` from the encoder
    /// - `position_offset`: index of the first token in `input_ids` relative to
    ///   the decoder start (0 for the first call). Used to slice the learned
    ///   positional embedding correctly for incremental decoding without a KV cache.
    pub fn forward_inference<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = input_ids.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Token embeddings: [B, T, hidden]
        let tok = self.embed_tokens.forward(client, input_ids)?;

        // Positional slice: [seq_len, hidden] from [max_target_positions, hidden]
        let pos = self
            .embed_positions
            .narrow(0, position_offset, seq_len)
            .map_err(Error::Numr)?;
        let pos = pos
            .reshape(&[1, seq_len, self.hidden_size])
            .map_err(Error::Numr)?;
        let pos_b = pos
            .broadcast_to(&[batch, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .contiguous()?;

        let mut hidden = client.add(tok.tensor(), &pos_b).map_err(Error::Numr)?;

        // Stack of decoder layers. Causal only matters for prefill (seq_len > 1);
        // for single-token incremental steps it's a no-op.
        for layer in &self.layers {
            hidden = layer.forward_inference(client, &hidden, encoder_out, true)?;
        }

        // Final LN
        let normed = self.layer_norm.forward(client, &Var::new(hidden, false))?;
        let hidden = normed.tensor().clone();

        // Logits = hidden @ embed_tokens.weight.T: [B, T, hidden] @ [hidden, vocab].
        // w_t stays a view so the matmul reads the weight transposed without a copy.
        let w_t = self.tied_out_weight.transpose(0, 1).map_err(Error::Numr)?;
        let logits = client.matmul(&hidden, &w_t).map_err(Error::Numr)?;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::VarMap;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const VOCAB: usize = 12;
    const MAX_TARGET: usize = 16;
    const LAYERS: usize = 2;

    fn cpu_setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    /// Deterministic pseudo-random fill, scaled small to keep the tiny model
    /// numerically stable across a few layers.
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_add(seed).wrapping_mul(2_654_435_761);
                ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn insert(
        varmap: &mut VarMap<CpuRuntime>,
        device: &CpuDevice,
        name: &str,
        shape: &[usize],
        seed: u32,
    ) {
        let data = fill(shape.iter().product(), seed);
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("build tensor");
        varmap.insert(name.to_string(), tensor);
    }

    fn insert_linear(
        varmap: &mut VarMap<CpuRuntime>,
        device: &CpuDevice,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        seed: u32,
    ) {
        insert(
            varmap,
            device,
            &format!("{prefix}.weight"),
            &[out_dim, in_dim],
            seed,
        );
        insert(
            varmap,
            device,
            &format!("{prefix}.bias"),
            &[out_dim],
            seed.wrapping_add(1),
        );
    }

    fn insert_layernorm(
        varmap: &mut VarMap<CpuRuntime>,
        device: &CpuDevice,
        prefix: &str,
        dim: usize,
    ) {
        varmap.insert(
            format!("{prefix}.weight"),
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; dim], &[dim], device)
                .expect("ln weight"),
        );
        varmap.insert(
            format!("{prefix}.bias"),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; dim], &[dim], device).expect("ln bias"),
        );
    }

    fn insert_attn(
        varmap: &mut VarMap<CpuRuntime>,
        device: &CpuDevice,
        prefix: &str,
        dim: usize,
        seed_base: u32,
    ) {
        insert_linear(
            varmap,
            device,
            &format!("{prefix}.q_proj"),
            dim,
            dim,
            seed_base,
        );
        insert_linear(
            varmap,
            device,
            &format!("{prefix}.k_proj"),
            dim,
            dim,
            seed_base + 10,
        );
        insert_linear(
            varmap,
            device,
            &format!("{prefix}.v_proj"),
            dim,
            dim,
            seed_base + 20,
        );
        insert_linear(
            varmap,
            device,
            &format!("{prefix}.out_proj"),
            dim,
            dim,
            seed_base + 30,
        );
    }

    /// A tiny decoder-only model, built directly against `WhisperDecoder`
    /// (no encoder), for tests that only exercise the decoder layers.
    fn tiny_decoder(device: &CpuDevice) -> WhisperDecoder<CpuRuntime> {
        let config = AudioConfig {
            encoder_type: "whisper".to_string(),
            hidden_size: HIDDEN,
            num_layers: 1,
            num_heads: HEADS,
            num_mel_bins: 4,
            max_audio_len: 8,
            projector_type: "linear".to_string(),
            vocab_size: VOCAB,
            decoder_layers: Some(LAYERS),
            max_target_positions: MAX_TARGET,
            intermediate_size: None,
        };
        let mut varmap = VarMap::<CpuRuntime>::new();
        let ffn = HIDDEN * 4;
        insert(
            &mut varmap,
            device,
            "embed_tokens.weight",
            &[VOCAB, HIDDEN],
            300,
        );
        insert(
            &mut varmap,
            device,
            "embed_positions.weight",
            &[MAX_TARGET, HIDDEN],
            301,
        );
        for i in 0..LAYERS {
            let base = 1000 + i as u32 * 100;
            let p = format!("layers.{i}");
            insert_attn(&mut varmap, device, &format!("{p}.self_attn"), HIDDEN, base);
            insert_layernorm(
                &mut varmap,
                device,
                &format!("{p}.self_attn_layer_norm"),
                HIDDEN,
            );
            insert_attn(
                &mut varmap,
                device,
                &format!("{p}.encoder_attn"),
                HIDDEN,
                base + 40,
            );
            insert_layernorm(
                &mut varmap,
                device,
                &format!("{p}.encoder_attn_layer_norm"),
                HIDDEN,
            );
            insert_linear(
                &mut varmap,
                device,
                &format!("{p}.fc1"),
                ffn,
                HIDDEN,
                base + 80,
            );
            insert_linear(
                &mut varmap,
                device,
                &format!("{p}.fc2"),
                HIDDEN,
                ffn,
                base + 85,
            );
            insert_layernorm(
                &mut varmap,
                device,
                &format!("{p}.final_layer_norm"),
                HIDDEN,
            );
        }
        insert_layernorm(&mut varmap, device, "layer_norm", HIDDEN);

        let mut vb = VarBuilder::new(&mut varmap, device);
        WhisperDecoder::from_varbuilder(&mut vb, &config).expect("build tiny decoder")
    }

    /// `forward_inference` (full recompute, no KV cache) and `forward_with_cache`
    /// (cache empty, prefill) must agree bit-for-bit on a shared prefix: this
    /// is the only test exercising the non-cached `self_attention` /
    /// `cross_attention` paths, and it pins them against the cached paths
    /// sharing the same `attend` core.
    #[test]
    fn forward_inference_matches_cached_prefill_logits() {
        let (client, device) = cpu_setup();
        let decoder = tiny_decoder(&device);
        let encoder_out =
            Tensor::<CpuRuntime>::from_slice(&fill(5 * HIDDEN, 42), &[1, 5, HIDDEN], &device)
                .expect("encoder_out");
        let ids: Vec<i64> = vec![1, 2, 3];
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &device).expect("input_ids");

        let uncached = decoder
            .forward_inference(&client, &input_ids, &encoder_out, 0)
            .expect("uncached forward");

        let mut cache = decoder.new_cache();
        let cached = decoder
            .forward_with_cache(&client, &input_ids, &encoder_out, 0, &mut cache)
            .expect("cached forward");

        let a = uncached.contiguous().unwrap().to_vec::<f32>();
        let b = cached.contiguous().unwrap().to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "logits mismatch: {x} vs {y}");
        }
    }
}

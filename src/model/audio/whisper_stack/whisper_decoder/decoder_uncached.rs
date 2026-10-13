//! `WhisperDecoder::forward_inference`: a full decoder pass that recomputes
//! every position without a KV cache.

use super::decoder::WhisperDecoder;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> WhisperDecoder<R> {
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
    use crate::model::config::AudioConfig;
    use crate::nn::{VarBuilder, VarMap};
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

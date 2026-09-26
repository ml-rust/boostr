//! Greedy autoregressive generation over a KV-cache-backed [`LoadedModel`].
//!
//! Prefills the prompt in one forward pass, then decodes one token at a
//! time: argmax the last position's logits, read that single token id back
//! to the host, feed it in as the next step's input. No other tensor
//! crosses the device boundary during decode.

use super::options::GenerateOptions;
use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredKvCache;
use crate::model::registry::LoadedModel;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Runs greedy generation from `prompt` and returns the generated token ids
/// (the prompt itself is not included).
///
/// Supports every `LoadedModel` kind whose `forward_with_kv_cache` works: the
/// Llama family and `Multimodal` wrapping one. Mamba (SSM state), `Hybrid`
/// and `Qwen35` (GDN state) do not use a flat KV cache and are refused with
/// an error naming the model kind — those need their own state type and a
/// dedicated loop, not this one.
///
/// Stops after `opts.max_new_tokens` tokens, when a chosen token is in
/// `opts.stop_tokens` (included in the result), or when the KV cache reaches
/// the model's `max_seq_len` (whichever comes first).
pub fn generate_greedy<R>(
    client: &R::Client,
    model: &LoadedModel<R>,
    prompt: &[u32],
    opts: &GenerateOptions,
) -> Result<Vec<u32>>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    if prompt.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "prompt",
            reason: "prompt has no tokens".into(),
        });
    }

    let num_kv_heads = model
        .num_kv_heads()
        .ok_or_else(|| no_kv_cache_error(model))?;
    let head_dim = model.head_dim().ok_or_else(|| no_kv_cache_error(model))?;
    let max_seq_len = model.max_seq_len();
    let device = client.device();

    let capacity = (prompt.len() + opts.max_new_tokens)
        .min(max_seq_len)
        .max(prompt.len());
    let mut kv_cache = LayeredKvCache::<R>::new_positional(
        model.num_layers(),
        1,
        num_kv_heads,
        capacity,
        max_seq_len,
        head_dim,
        model.dtype(),
        device,
    )?;

    let prompt_ids: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
    let input =
        Tensor::<R>::from_slice(&prompt_ids, &[1, prompt.len()], device).map_err(Error::Numr)?;
    let mut logits = model.forward_with_kv_cache(&input, &mut kv_cache, 0)?;

    let mut generated = Vec::with_capacity(opts.max_new_tokens);
    for _ in 0..opts.max_new_tokens {
        let next_token = argmax_last_token::<R>(&logits)?;
        generated.push(next_token);
        if opts.stop_tokens.contains(&next_token) {
            break;
        }

        let position = kv_cache.seq_len();
        if position >= max_seq_len {
            break;
        }

        let next_input =
            Tensor::<R>::from_slice(&[next_token as i64], &[1, 1], device).map_err(Error::Numr)?;
        logits = model.forward_with_kv_cache(&next_input, &mut kv_cache, position)?;
    }

    Ok(generated)
}

/// Argmax over the vocab dim of the last position's logits, read back as a
/// single token id. The only device→host transfer this step makes.
fn argmax_last_token<R>(logits: &Tensor<R>) -> Result<u32>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let seq_len = logits.shape()[1];
    let last = logits.narrow(1, seq_len - 1, 1).map_err(Error::Numr)?;
    let vocab_dim = last.shape().len() - 1;
    let ids = last.argmax(vocab_dim, false).map_err(Error::Numr)?;
    let scalar = ids.squeeze(None);
    let id: i64 = scalar.item().map_err(Error::Numr)?;
    Ok(id as u32)
}

fn no_kv_cache_error<R: Runtime<DType = DType>>(model: &LoadedModel<R>) -> Error
where
    R::Client: ModelClient<R>,
{
    Error::ModelError {
        reason: format!(
            "{} has no flat KV-cache layout — generate_greedy needs forward_with_kv_cache, \
             which this model kind does not support",
            model.model_type()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use crate::model::llama::Llama;
    use crate::model::traits::Model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn tiny_config() -> ModelConfig {
        let yaml = r#"
model_type: llama
vocab_size: 32
hidden_size: 16
num_layers: 2
max_seq_len: 32
intermediate_size: 32
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  rope_theta: 10000.0
"#;
        serde_saphyr::from_str(yaml).unwrap()
    }

    fn tiny_model() -> (
        numr::runtime::cpu::CpuClient,
        numr::runtime::cpu::CpuDevice,
        LoadedModel<CpuRuntime>,
    ) {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        (client, device, LoadedModel::Llama(Box::new(model)))
    }

    /// Recomputes the whole sequence from scratch at every step (no KV
    /// cache) and takes the argmax of the last position — the ground truth
    /// `generate_greedy` must match token for token.
    fn naive_greedy(
        client: &numr::runtime::cpu::CpuClient,
        model: &LoadedModel<CpuRuntime>,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Vec<u32> {
        let device = client.device();
        let mut ids: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
        let mut generated = Vec::new();
        for _ in 0..max_new_tokens {
            let input = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], device).unwrap();
            let mut kv_cache = LayeredKvCache::<CpuRuntime>::new_positional(
                model.num_layers(),
                1,
                model.num_kv_heads().unwrap(),
                ids.len(),
                model.max_seq_len(),
                model.head_dim().unwrap(),
                DType::F32,
                device,
            )
            .unwrap();
            let logits = model
                .forward_with_kv_cache(&input, &mut kv_cache, 0)
                .unwrap();
            let seq_len = logits.shape()[1];
            let last = logits.narrow(1, seq_len - 1, 1).unwrap();
            let ids_tensor = last.argmax(2, false).unwrap();
            let scalar = ids_tensor.squeeze(None);
            let next: i64 = scalar.item().unwrap();
            let next = next as u32;
            generated.push(next);
            ids.push(next as i64);
        }
        generated
    }

    #[test]
    fn generate_greedy_matches_naive_full_recompute_loop() {
        let (client, _device, model) = tiny_model();
        let prompt = [1u32, 2, 3];
        let opts = GenerateOptions {
            max_new_tokens: 5,
            stop_tokens: Vec::new(),
        };

        let via_kv_cache = generate_greedy(&client, &model, &prompt, &opts).unwrap();
        let via_naive = naive_greedy(&client, &model, &prompt, 5);

        assert_eq!(via_kv_cache, via_naive);
    }

    #[test]
    fn generate_greedy_stops_at_a_stop_token() {
        let (client, _device, model) = tiny_model();
        let prompt = [1u32, 2, 3];

        // Run once, unconstrained, to learn what the first generated token is.
        let opts = GenerateOptions {
            max_new_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let first = generate_greedy(&client, &model, &prompt, &opts).unwrap();
        let first_token = first[0];

        let opts = GenerateOptions {
            max_new_tokens: 10,
            stop_tokens: vec![first_token],
        };
        let out = generate_greedy(&client, &model, &prompt, &opts).unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0], first_token);
    }

    #[test]
    fn generate_greedy_respects_max_new_tokens() {
        let (client, _device, model) = tiny_model();
        let prompt = [1u32, 2, 3];
        let opts = GenerateOptions {
            max_new_tokens: 4,
            stop_tokens: Vec::new(),
        };

        let out = generate_greedy(&client, &model, &prompt, &opts).unwrap();

        assert_eq!(out.len(), 4);
    }

    #[test]
    fn generate_greedy_refuses_a_model_kind_without_a_flat_kv_cache() {
        let (client, device) = cpu_setup();
        let qwen35 = crate::model::qwen35::model::tiny_model(&device, 0x3535_0200);
        let model = LoadedModel::Qwen35(Box::new(qwen35));
        let opts = GenerateOptions::default();

        let err = generate_greedy(&client, &model, &[1u32, 2], &opts).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("qwen35"),
            "error should name the model kind: {msg}"
        );
    }
}

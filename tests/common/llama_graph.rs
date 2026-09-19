//! Shared CUDA graph-mode decode helpers for Llama-family models: the
//! `Dims` a decode needs, eager/graph decode over a `LoadedModel`, and
//! loading a real checkpoint (GGUF or SafeTensors directory) the way
//! blazr's loader does.

use std::path::Path;

use boostr::format::{Gguf, GgufMetadata, SafeTensorsLoader};
use boostr::inference::LayeredKvCache;
use boostr::inference::decode_graph::{
    DecodeGraph, DeviceScalars, argmax_to_buf, copy_into_stable,
};
use boostr::model::config::{AttentionConfig, UniversalConfig};
use boostr::model::{LoadedModel, load_huggingface_config};
use boostr::nn::{VarBuilder, VarMap};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Model dimensions a decode needs: layer count, KV heads, head dim, vocab.
#[derive(Clone, Copy)]
pub struct Dims {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
}

impl Dims {
    /// Read the dims off a loaded model's accessors, for an attention model
    /// (a KV cache needs `num_kv_heads`/`head_dim`, both `None` for an SSM).
    pub fn from_model(model: &LoadedModel<CudaRuntime>) -> Self {
        Self {
            layers: model.num_layers(),
            kv_heads: model.num_kv_heads().expect("model has attention KV heads"),
            head_dim: model.head_dim().expect("model has attention head dim"),
            vocab: model.vocab_size(),
        }
    }
}

pub fn row_f32(client: &CudaClient, logits: &Tensor<CudaRuntime>) -> Vec<f32> {
    if logits.dtype() == DType::F32 {
        logits.to_vec::<f32>()
    } else {
        client.cast(logits, DType::F32).unwrap().to_vec::<f32>()
    }
}

pub fn full_cache(
    device: &CudaDevice,
    capacity: usize,
    dtype: DType,
    dims: &Dims,
) -> LayeredKvCache<CudaRuntime> {
    LayeredKvCache::<CudaRuntime>::new_positional(
        dims.layers,
        1,
        dims.kv_heads,
        capacity,
        capacity,
        dims.head_dim,
        dtype,
        device,
    )
    .unwrap()
}

/// Eager prefill of `prompt` at position 0; returns the last logits row.
pub fn prefill(
    client: &CudaClient,
    device: &CudaDevice,
    model: &LoadedModel<CudaRuntime>,
    kv: &mut LayeredKvCache<CudaRuntime>,
    prompt: &[i64],
    vocab: usize,
) -> Vec<f32> {
    let ids = Tensor::<CudaRuntime>::from_slice(prompt, &[1, prompt.len()], device).unwrap();
    let logits = model.forward_with_kv_cache(&ids, kv, 0).unwrap();
    let flat = row_f32(client, &logits);
    flat[(prompt.len() - 1) * vocab..].to_vec()
}

/// Eager greedy decode: `steps` logits rows and the ids they produce.
pub fn eager_decode(
    client: &CudaClient,
    device: &CudaDevice,
    model: &LoadedModel<CudaRuntime>,
    first: i64,
    steps: usize,
    kv: &mut LayeredKvCache<CudaRuntime>,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let mut token = first;
    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for _ in 0..steps {
        let position = kv.seq_len();
        let x = Tensor::<CudaRuntime>::from_slice(&[token], &[1, 1], device).unwrap();
        let logits = model.forward_with_kv_cache(&x, kv, position).unwrap();
        let row = row_f32(client, &logits);
        token = super::qwen35_cuda::argmax(&row);
        rows.push(row);
        ids.push(token);
    }
    (rows, ids)
}

/// Graph greedy decode, built the way a serving session builds it. Returns
/// the re-prefill's first id, then per-step logits rows and ids.
#[allow(clippy::too_many_arguments)]
pub fn graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    model: &LoadedModel<CudaRuntime>,
    dims: &Dims,
    prompt: &[i64],
    capacity: usize,
    dtype: DType,
    steps: usize,
) -> (i64, Vec<Vec<f32>>, Vec<i64>) {
    let half_dim = dims.head_dim / 2;
    let mut kv = full_cache(device, capacity, dtype, dims);

    // First prefill: fills the cache the warmup pass reads.
    let _ = prefill(client, device, model, &mut kv, prompt, dims.vocab);

    let (rope_cos_var, rope_sin_var) = model.rope_caches().unwrap();
    let rope_cos_cache = rope_cos_var.tensor().clone();
    let rope_sin_cache = rope_sin_var.tensor().clone();

    // Stable-address decode inputs, all allocated before capture.
    let token_buf = Tensor::<CudaRuntime>::zeros(&[1, 1], DType::I64, device).unwrap();
    let cos_slice = Var::new(
        Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_cos_cache.dtype(), device).unwrap(),
        false,
    );
    let sin_slice = Var::new(
        Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_sin_cache.dtype(), device).unwrap(),
        false,
    );
    let device_scalars = DeviceScalars::new(kv.seq_len(), device).unwrap();
    let next_token_buf = Tensor::<CudaRuntime>::zeros(&[1], DType::I64, device).unwrap();
    let logits_buf = Tensor::<CudaRuntime>::zeros(&[1, 1, dims.vocab], dtype, device).unwrap();

    // Warmup pass: every kernel loads, nothing captured.
    device_scalars.update(client, kv.seq_len()).unwrap();
    let warmup_logits = model
        .forward_graph_mode(&token_buf, &mut kv, &device_scalars, &cos_slice, &sin_slice)
        .unwrap();
    argmax_to_buf(client, &warmup_logits, &next_token_buf).unwrap();
    drop(warmup_logits);

    // The warmup wrote the cache: reset and prefill again at the same addresses.
    kv.reset();
    let prefill_row = prefill(client, device, model, &mut kv, prompt, dims.vocab);
    let first = super::qwen35_cuda::argmax(&prefill_row);

    let capture_seq_len = kv.seq_len();
    device_scalars.update(client, capture_seq_len).unwrap();
    device_scalars
        .update_rope_slices(
            client,
            &rope_cos_cache,
            &rope_sin_cache,
            &cos_slice,
            &sin_slice,
            capture_seq_len,
            half_dim,
        )
        .unwrap();
    // The slice must hold row `capture_seq_len` of the table at the table's
    // dtype; a byte-size mismatch reads the wrong row.
    let expect = rope_cos_cache.narrow(0, capture_seq_len, 1).unwrap();
    assert_eq!(
        row_f32(client, cos_slice.tensor()),
        row_f32(client, &expect),
        "{dtype:?}: cos slice differs from table row {capture_seq_len}"
    );

    let graph = CudaRuntime::capture_graph_into(client, &[&token_buf], &[&next_token_buf], |c| {
        let logits = model
            .forward_graph_mode(&token_buf, &mut kv, &device_scalars, &cos_slice, &sin_slice)
            .map_err(|e| numr::error::Error::Backend(format!("capture forward: {e}")))?;
        copy_into_stable(c, &logits, &logits_buf)?;
        argmax_to_buf(c, &logits, &next_token_buf)
    })
    .unwrap();

    let mut decode = DecodeGraph {
        graph,
        device_scalars,
        token_buf,
        cos_slice: cos_slice.tensor().clone(),
        sin_slice: sin_slice.tensor().clone(),
        rope_cos_cache,
        rope_sin_cache,
        next_token_buf,
        head_dim: half_dim,
        seq_len: capture_seq_len,
    };
    decode.seed_next_token(client, first).unwrap();

    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for _ in 0..steps {
        decode.pre_replay_and_launch(client).unwrap();
        ids.push(decode.next_token_buf.to_vec::<i64>()[0]);
        rows.push(row_f32(client, &logits_buf));
    }
    (first, rows, ids)
}

/// Load a real checkpoint the way blazr's loader does: a GGUF file through
/// `Gguf`/`VarMap::from_gguf`, or a SafeTensors directory through its
/// `config.json` and `SafeTensorsLoader`.
pub fn load_real_model(
    path: &Path,
    device: &CudaDevice,
) -> Result<LoadedModel<CudaRuntime>, String> {
    if path.is_dir() {
        load_safetensors_dir(path, device)
    } else {
        load_gguf_file(path, device)
    }
}

fn load_gguf_file(path: &Path, device: &CudaDevice) -> Result<LoadedModel<CudaRuntime>, String> {
    let gguf = Gguf::open(path).map_err(|e| format!("open gguf: {e}"))?;
    let config = llama_config_from_gguf(gguf.metadata())?;
    drop(gguf);
    let mut varmap = VarMap::<CudaRuntime>::from_gguf(path, device)
        .map_err(|e| format!("load gguf tensors: {e}"))?;
    let mut vb = VarBuilder::new(&mut varmap, device);
    LoadedModel::<CudaRuntime>::load(&config, &mut vb).map_err(|e| format!("load model: {e}"))
}

fn load_safetensors_dir(
    dir: &Path,
    device: &CudaDevice,
) -> Result<LoadedModel<CudaRuntime>, String> {
    let config = load_huggingface_config(dir.join("config.json"))
        .map_err(|e| format!("load config.json: {e}"))?;
    let mut loader = SafeTensorsLoader::open(dir).map_err(|e| format!("open safetensors: {e}"))?;
    let mut varmap = VarMap::<CudaRuntime>::new();
    for name in loader.tensor_names() {
        let tensor = loader
            .load_tensor::<CudaRuntime>(&name, device)
            .map_err(|e| format!("load tensor {name}: {e}"))?;
        varmap.insert(name, tensor);
    }
    let mut vb = VarBuilder::new(&mut varmap, device);
    LoadedModel::<CudaRuntime>::load(&config, &mut vb).map_err(|e| format!("load model: {e}"))
}

/// A Llama-family `UniversalConfig` from GGUF metadata: vocab, hidden size,
/// layer count and attention shape, off boostr's own metadata accessors.
/// Covers `llama`/`mistral`-shaped GGUFs; architecture-specific key tables
/// (MoE, SSM, ALiBi, …) live in blazr's loader, not here.
fn llama_config_from_gguf(metadata: &GgufMetadata) -> Result<UniversalConfig, String> {
    let arch = metadata
        .architecture()
        .ok_or("gguf missing general.architecture")?;
    let hidden_size = metadata
        .embedding_length()
        .ok_or("gguf missing embedding_length")? as usize;
    let num_layers = metadata.block_count().ok_or("gguf missing block_count")? as usize;
    let max_seq_len = metadata
        .context_length()
        .map(|v| v as usize)
        .unwrap_or(4096);
    let vocab_size = metadata
        .get_u32("general.vocab_size")
        .map(|v| v as usize)
        .or_else(|| metadata.get_array("tokenizer.ggml.tokens").map(|a| a.len()))
        .ok_or("gguf missing vocab size")?;
    let num_heads = metadata
        .get_u32(&format!("{arch}.attention.head_count"))
        .ok_or("gguf missing attention.head_count")? as usize;
    let num_kv_heads = metadata
        .get_u32(&format!("{arch}.attention.head_count_kv"))
        .map(|v| v as usize);
    let head_dim = metadata
        .get_u32(&format!("{arch}.attention.key_length"))
        .map(|v| v as usize)
        .or_else(|| hidden_size.checked_div(num_heads));
    let rope_theta = metadata
        .get_f32(&format!("{arch}.rope.freq_base"))
        .unwrap_or(10_000.0);
    let intermediate_size = metadata
        .get_u32(&format!("{arch}.feed_forward_length"))
        .map(|v| v as usize);
    let rms_norm_eps = metadata
        .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .map(|v| v as f64)
        .unwrap_or(1e-5);

    Ok(UniversalConfig {
        model_type: "llama".into(),
        vocab_size,
        hidden_size,
        num_layers,
        max_seq_len,
        intermediate_size,
        rms_norm_eps,
        attention: Some(AttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_scaling: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            sliding_window: None,
            use_alibi: false,
        }),
        ssm: None,
        moe: None,
        hybrid_layers: None,
        tie_word_embeddings: false,
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: None,
        qwen35_attention: None,
        hadamard: None,
        quant_formats: Vec::new(),
    })
}

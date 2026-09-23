//! CUDA harness shared by the `qwen35` graph-mode tests: full-capacity
//! caches, eager prefill and greedy decode, and flat state readbacks.

use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::qwen35::Qwen35Model;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

pub fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

pub fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

/// Full-capacity one-lane KV cache (graph mode needs stable k/v addresses)
/// plus zero GDN state.
pub fn caches(
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    capacity: usize,
) -> (LayeredKvCache<CudaRuntime>, LayeredGdnState<CudaRuntime>) {
    caches_batched(device, model, capacity, 1)
}

/// The same containers with `batch` lanes, one per sequence in flight.
pub fn caches_batched(
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    capacity: usize,
    batch: usize,
) -> (LayeredKvCache<CudaRuntime>, LayeredGdnState<CudaRuntime>) {
    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: batch,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: capacity,
        max_seq_len: capacity,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::<CudaRuntime>::new(model.num_attention_layers(), &kv_config, device)
        .unwrap();
    let gdn = LayeredGdnState::<CudaRuntime>::zeros(
        model.num_gdn_layers(),
        model.gdn_config(),
        batch,
        DType::F32,
        device,
    )
    .unwrap();
    (kv, gdn)
}

pub fn argmax(row: &[f32]) -> i64 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best as i64
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Eager prefill of `prompt`; returns the argmax of the last row.
pub fn prefill(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    prompt: &[i64],
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
) -> i64 {
    let ids = Tensor::<CudaRuntime>::from_slice(prompt, &[1, prompt.len()], device).unwrap();
    let logits = model.forward_qwen35(client, &ids, kv, gdn, 0).unwrap();
    let vocab = logits.shape()[2];
    let flat = logits.to_vec::<f32>();
    argmax(&flat[(prompt.len() - 1) * vocab..])
}

/// Eager greedy decode: `steps` rows of logits and the ids they produce.
pub fn eager_decode(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    first: i64,
    steps: usize,
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let mut token = first;
    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for _ in 0..steps {
        let position = kv.seq_len();
        let x = Tensor::<CudaRuntime>::from_slice(&[token], &[1, 1], device).unwrap();
        let logits = model.forward_qwen35(client, &x, kv, gdn, position).unwrap();
        let row = logits.to_vec::<f32>();
        token = argmax(&row);
        rows.push(row);
        ids.push(token);
    }
    (rows, ids)
}

/// Every layer's conv window and ssm state, concatenated.
pub fn gdn_flat(state: &LayeredGdnState<CudaRuntime>) -> (Vec<f32>, Vec<f32>) {
    let mut conv = Vec::new();
    let mut ssm = Vec::new();
    for i in 0..state.num_layers() {
        let layer = state.layer(i).unwrap();
        conv.extend(layer.conv().to_vec::<f32>());
        ssm.extend(layer.ssm().to_vec::<f32>());
    }
    (conv, ssm)
}

/// Device addresses of every KV k/v buffer and every GDN conv/ssm buffer.
pub fn state_ptrs(
    kv: &LayeredKvCache<CudaRuntime>,
    gdn: &LayeredGdnState<CudaRuntime>,
) -> Vec<u64> {
    let mut ptrs = Vec::new();
    for i in 0..kv.num_layers() {
        let layer = kv.layer(i).unwrap();
        ptrs.push(layer.k_cache_raw().ptr());
        ptrs.push(layer.v_cache_raw().ptr());
    }
    for i in 0..gdn.num_layers() {
        let layer = gdn.layer(i).unwrap();
        ptrs.push(layer.conv().ptr());
        ptrs.push(layer.ssm().ptr());
    }
    ptrs
}

//! Shared fixtures for the paged-attention unit tests: an identity-weight
//! attention block, a RoPE cache, paged caches, slot mappings, inputs, and
//! the dense causal reference.

use super::super::LlamaAttention;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::nn::{Linear, MaybeQuantLinear, RoPE};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(super) const HIDDEN: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
pub(super) const BLOCK_SIZE: usize = 4;

/// `[n, n]` identity, so `Linear::forward` is a pass-through and Q/K/V
/// equal the (reshaped) input — the test then only exercises RoPE and
/// the paged-cache plumbing, not arbitrary projection weights.
fn identity_weight(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::<CpuRuntime>::from_slice(&data, &[n, n], device).unwrap()
}

pub(super) fn test_attention(device: &CpuDevice) -> LlamaAttention<CpuRuntime> {
    let identity =
        || MaybeQuantLinear::Standard(Linear::new(identity_weight(HIDDEN, device), None, false));
    LlamaAttention {
        q_proj: identity(),
        k_proj: identity(),
        v_proj: identity(),
        o_proj: identity(),
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        q_norm: None,
        k_norm: None,
        use_alibi: false,
        sliding_window: 0,
        softcap: None,
        clip_qkv: None,
        alibi_scale: 1.0,
    }
}

pub(super) fn test_rope(device: &CpuDevice) -> RoPE<CpuRuntime> {
    RoPE::<CpuRuntime>::precompute_freqs(128, HEAD_DIM, 10000.0, None, device)
        .expect("rope cache build must succeed")
}

/// A fresh single-layer paged cache with one block per row, so each
/// row's freshly written token is the only key it can attend.
pub(super) fn test_cache(batch: usize, device: &CpuDevice) -> LayeredPagedKvCache<CpuRuntime> {
    LayeredPagedKvCache::<CpuRuntime>::new(
        1,
        batch,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        DType::F32,
        device,
    )
    .expect("paged cache build must succeed")
}

/// Row `b` writes into physical block `b`, slot `b * BLOCK_SIZE`.
pub(super) fn slot_mapping_and_block_table(
    batch: usize,
    device: &CpuDevice,
) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
    let slots: Vec<i32> = (0..batch).map(|b| (b * BLOCK_SIZE) as i32).collect();
    let slot_mapping = Tensor::<CpuRuntime>::from_slice(&slots, &[batch], device).unwrap();
    let bt: Vec<i32> = (0..batch as i32).collect();
    let block_table = Tensor::<CpuRuntime>::from_slice(&bt, &[batch, 1], device).unwrap();
    (slot_mapping, block_table)
}

/// Deterministic, distinct-per-row decode input `[batch, 1, HIDDEN]`.
pub(super) fn decode_input(batch: usize, device: &CpuDevice) -> Var<CpuRuntime> {
    let data: Vec<f32> = (0..batch * HIDDEN)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[batch, 1, HIDDEN], device).unwrap(),
        false,
    )
}

/// Runs one paged decode step and returns the flat `f32` output.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_decode(
    attn: &LlamaAttention<CpuRuntime>,
    rope: &RoPE<CpuRuntime>,
    x: &Var<CpuRuntime>,
    batch: usize,
    position: usize,
    rope_positions: Option<&Tensor<CpuRuntime>>,
    device: &CpuDevice,
) -> Vec<f32> {
    let (client, _) = crate::test_utils::cpu_setup();
    let cache = test_cache(batch, device);
    let (slot_mapping, block_table) = slot_mapping_and_block_table(batch, device);

    let out = attn
        .forward_with_paged_kv_cache(
            &client,
            x,
            rope,
            &cache,
            0,
            &slot_mapping,
            &block_table,
            1,
            position,
            None,
            rope_positions,
        )
        .expect("paged decode forward must succeed");
    out.tensor().contiguous().unwrap().to_vec::<f32>()
}

/// Deterministic prefill input `[1, seq_len, HIDDEN]`, distinct per
/// position so a masking or offset bug cannot hide behind repeated rows.
pub(super) fn prefill_input(seq_len: usize, device: &CpuDevice) -> Var<CpuRuntime> {
    let data: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| (i as f32 * 0.11).cos())
        .collect();
    Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq_len, HIDDEN], device).unwrap(),
        false,
    )
}

/// A fresh single-layer, single-row paged cache with `num_blocks`
/// physical blocks, for prefill tests that write more than one block
/// worth of tokens.
pub(super) fn prefill_cache(
    num_blocks: usize,
    device: &CpuDevice,
) -> LayeredPagedKvCache<CpuRuntime> {
    LayeredPagedKvCache::<CpuRuntime>::new(
        1,
        num_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        DType::F32,
        device,
    )
    .expect("paged cache build must succeed")
}

/// Slot mapping for absolute token positions `start..start+count`, and
/// the fixed block table `[0, 1, ..., num_blocks-1]` (physical block id
/// equals logical block index). Row 0 only.
pub(super) fn prefill_slots(
    start: usize,
    count: usize,
    num_blocks: usize,
    device: &CpuDevice,
) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
    let slots: Vec<i32> = (start..start + count)
        .map(|i| {
            let block_idx = i / BLOCK_SIZE;
            let offset = i % BLOCK_SIZE;
            (block_idx * BLOCK_SIZE + offset) as i32
        })
        .collect();
    let slot_mapping = Tensor::<CpuRuntime>::from_slice(&slots, &[count], device).unwrap();
    let bt: Vec<i32> = (0..num_blocks as i32).collect();
    let block_table = Tensor::<CpuRuntime>::from_slice(&bt, &[1, num_blocks], device).unwrap();
    (slot_mapping, block_table)
}

/// Runs `x` through the dense, non-paged path (`forward_with_kv_cache`)
/// over a fresh contiguous KV cache sized for exactly `seq_len` tokens,
/// and returns the flat `f32` output. This is the causal-masking
/// reference the paged-prefill tests diff against: an unrelated kernel
/// and cache layout from `paged_attention_fwd`, so agreement does not
/// depend on the two paths sharing a masking bug.
pub(super) fn dense_reference_output(
    client: &CpuClient,
    attn: &LlamaAttention<CpuRuntime>,
    rope: &RoPE<CpuRuntime>,
    x: &Var<CpuRuntime>,
    seq_len: usize,
    device: &CpuDevice,
) -> Vec<f32> {
    let mut dense_cache = crate::inference::KvCache::<CpuRuntime>::new(
        1,
        NUM_KV_HEADS,
        seq_len,
        seq_len,
        HEAD_DIM,
        DType::F32,
        device,
    )
    .expect("dense kv cache build must succeed");
    attn.forward_with_kv_cache(client, x, rope, &mut dense_cache, 0)
        .expect("dense prefill must succeed")
        .tensor()
        .contiguous()
        .unwrap()
        .to_vec::<f32>()
}

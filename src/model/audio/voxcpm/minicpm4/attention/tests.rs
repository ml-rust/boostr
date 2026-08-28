//! Split out of `attention.rs` to keep that file under the crate's 500-line
//! hard limit for model-architecture files after `MiniCpm4Attention::apply_lora`
//! was added. `use super::*;` below reaches every item `attention.rs` itself
//! imported, exactly as if this module were still inline.

use super::*;
use crate::nn::Weight;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const HIDDEN: usize = 4;
const NUM_HEADS: usize = 1;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;

/// Deterministic, non-degenerate weights: zeros would make every
/// assertion below pass vacuously.
fn filled(shape: &[usize], salt: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 29 + salt * 7) % 11) as f32 - 5.0) / 8.0)
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("weights")
}

fn tiny_attention(no_rope: bool, device: &CpuDevice) -> MiniCpm4Attention<CpuRuntime> {
    let linear = |salt| -> MaybeLoraLinear<CpuRuntime> {
        MaybeQuantLinear::from_weight(
            Weight::Standard(filled(&[HIDDEN, HIDDEN], salt, device)),
            None,
        )
        .into()
    };
    MiniCpm4Attention {
        q_proj: linear(1),
        k_proj: linear(2),
        v_proj: linear(3),
        o_proj: linear(4),
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        no_rope,
    }
}

/// One `[1, 1, HIDDEN]` embedding.
fn embed(salt: usize, device: &CpuDevice) -> Var<CpuRuntime> {
    Var::new(filled(&[1, 1, HIDDEN], salt, device), false)
}

/// The load-bearing property of NoPE: the block carries NO positional
/// signal, so the same embedding attending the same key set produces the
/// same output whatever absolute position it claims.
///
/// Both runs write one prior key/value into the cache, then present the
/// SAME query embedding at a different absolute position. The key set and
/// the causal mask are identical across the two runs (the mask is built
/// from the cache length, not from `position`), so the rotation is the
/// only thing that can differ — which is why the rotary half of this test
/// must, and does, disagree.
#[test]
fn nope_output_is_independent_of_absolute_position() {
    let (client, device) = cpu_setup();
    let rope =
        RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, &device).expect("rope");

    let run = |no_rope: bool, position: usize| {
        let attn = tiny_attention(no_rope, &device);
        let table = (!no_rope).then_some(&rope);
        let mut cache =
            KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
                .expect("cache");
        attn.forward_cached(&client, &embed(1, &device), table, &mut cache, 0)
            .expect("prior position");
        let out = attn
            .forward_cached(&client, &embed(2, &device), table, &mut cache, position)
            .expect("query position");
        out.tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>()
    };

    let near = run(true, 1);
    let far = run(true, 9);
    assert!(
        near.iter().any(|v| v.abs() > 1e-6),
        "degenerate output: the comparison below would pass vacuously"
    );
    for (a, b) in near.iter().zip(&far) {
        assert!(
            (a - b).abs() < 1e-6,
            "no_rope leaked a positional signal: {a} vs {b}"
        );
    }

    let rotary_near = run(false, 1);
    let rotary_far = run(false, 9);
    assert!(
        rotary_near
            .iter()
            .zip(&rotary_far)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "the rotary block was position-invariant too, so the NoPE half of \
         this test proves nothing"
    );
}

/// A block that rotates must not fall back to an unrotated forward when
/// the table is missing. Both paths error, neither panics.
#[test]
fn rotating_block_rejects_a_missing_rope_table() {
    let (client, device) = cpu_setup();
    let attn = tiny_attention(false, &device);

    let err = attn.forward(&client, &embed(1, &device), None).unwrap_err();
    assert!(err.to_string().contains("no_rope"), "got {err}");

    let mut cache =
        KvCache::<CpuRuntime>::new(1, NUM_KV_HEADS, 4, 4, HEAD_DIM, DType::F32, &device)
            .expect("cache");
    let err = attn
        .forward_cached(&client, &embed(1, &device), None, &mut cache, 0)
        .unwrap_err();
    assert!(err.to_string().contains("no_rope"), "got {err}");
    assert_eq!(cache.seq_len(), 0, "cache was written on the error path");
}

/// A NoPE block runs to completion with no table at all.
#[test]
fn nope_block_runs_without_a_table() {
    let (client, device) = cpu_setup();
    let attn = tiny_attention(true, &device);
    let out = attn
        .forward(&client, &embed(1, &device), None)
        .expect("forward");
    assert_eq!(out.shape(), &[1, 1, HIDDEN]);
}

/// `apply_lora` on a leaf attention block: matched targets get wrapped,
/// unmatched fields stay `Plain`, and the count reflects exactly the
/// matched set.
#[test]
fn apply_lora_wraps_only_targeted_projections() {
    let (_client, device) = cpu_setup();
    let mut attn = tiny_attention(false, &device);
    let targets = LoraTargets::new(["q_proj", "v_proj"]);

    let adapted = attn
        .apply_lora(&targets, 2, 4.0, &device, "self_attn")
        .expect("apply_lora");
    assert_eq!(adapted, 2);
    assert!(attn.q_proj.is_adapted());
    assert!(attn.v_proj.is_adapted());
    assert!(!attn.k_proj.is_adapted());
    assert!(!attn.o_proj.is_adapted());
}

/// Adapting an already-adapted projection errors rather than silently
/// discarding the existing adapter.
#[test]
fn apply_lora_rejects_double_adapt() {
    let (_client, device) = cpu_setup();
    let mut attn = tiny_attention(false, &device);
    let targets = LoraTargets::new(["q_proj"]);

    attn.apply_lora(&targets, 2, 4.0, &device, "self_attn")
        .expect("first apply_lora");
    let err = attn
        .apply_lora(&targets, 2, 4.0, &device, "self_attn")
        .unwrap_err();
    assert!(err.to_string().contains("already carries"), "got {err}");
}

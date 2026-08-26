//! Tests for the attention-core prologue/epilogue and RoPE-table validation.

use super::entry::attention_core_masked;
use super::spec::{AttentionCoreSpec, AttentionKernel};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// A rotating block (`use_alibi` and `skip_rope` both false) given `None`
/// RoPE tables must error, never silently skip the rotation or panic.
#[test]
fn rotating_block_rejects_none_rope_tables() {
    let (client, device) = cpu_setup();
    let shape = [1usize, 2, 4]; // [B, S, H*D] with num_heads=1, head_dim=4
    let data = vec![0.1f32; 8];
    let q = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &shape, &device).expect("q"),
        false,
    );
    let k = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &shape, &device).expect("k"),
        false,
    );
    let v = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &shape, &device).expect("v"),
        false,
    );

    let spec = AttentionCoreSpec::<CpuRuntime> {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        q_norm: None,
        k_norm: None,
        use_alibi: false,
        skip_rope: false,
        sliding_window: 0,
        kernel: AttentionKernel::Masked,
    };

    let err = attention_core_masked(&client, &q, &k, &v, None, None, &spec)
        .expect_err("a rotating block with None RoPE tables must error, not run");
    let text = err.to_string();
    assert!(
        text.contains("cos") || text.contains("sin") || text.contains("RoPE"),
        "error must name the missing RoPE tables, got: {text}"
    );
}

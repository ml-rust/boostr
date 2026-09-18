//! `kv_start` (per-row left padding) on the WebGPU `flash_attention_fwd`
//! (`shaders/attention/flash.wgsl`), against the sliced per-row reference in
//! `tests/common/kv_start.rs` run on the same WGPU client, and against the
//! CPU path for cross-backend agreement.
//!
//! The shader marks a row with no valid key with `lse = -1e30`, not `-inf`;
//! that sentinel is normalized to `-inf` before comparing.
//!
//! Run with:
//!   cd boostr && cargo test --features wgpu --test flash_kv_start_wgpu

#![cfg(feature = "wgpu")]

mod common;

use std::sync::{Mutex, OnceLock};

use boostr::ops::AttnOutLayout;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use common::cpu_setup;
use common::kv_start::{
    KvStartGeom, max_abs_diff, read_f32, sliced_reference, token_major_to_head_major, values,
};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

static WGPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn wgpu_setup() -> Option<(WgpuClient, WgpuDevice)> {
    let device = WgpuDevice::new(0);
    match WgpuClient::new(device.clone()) {
        Ok(client) => Some((client, device)),
        Err(e) => {
            eprintln!("WgpuClient::new failed ({e:?}); nothing verified");
            None
        }
    }
}

const STARTS: [i32; 3] = [0, 5, 17];

/// Bit identity, element by element, naming the first mismatch: the shader
/// walks each query's keys from its row's start in order, so a padded row
/// forms the bits its unpadded run does.
fn assert_bits(got: &[f32], want: &[f32], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{tag}: element {i} is {:e} ({:#010x}), reference {:e} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

fn geom(seq_q: usize, seq_k: usize, head_dim: usize, causal: bool, window: usize) -> KvStartGeom {
    KvStartGeom {
        batch: 3,
        num_heads: 4,
        num_kv_heads: 2,
        seq_q,
        seq_k,
        head_dim,
        causal,
        window,
    }
}

fn fixture_data(g: &KvStartGeom) -> [Vec<f32>; 3] {
    let q_n = g.batch * g.num_heads * g.seq_q * g.head_dim;
    let kv_n = g.batch * g.num_kv_heads * g.seq_k * g.head_dim;
    [values(q_n, 0.1), values(kv_n, 1.3), values(kv_n, 2.7)]
}

fn run(
    client: &WgpuClient,
    [q, k, v]: &[Tensor<WgpuRuntime>; 3],
    g: &KvStartGeom,
    kv_start: Option<&Tensor<WgpuRuntime>>,
    layout: AttnOutLayout,
) -> (Vec<f32>, Vec<f32>) {
    let (out, lse) = client
        .flash_attention_fwd(
            q,
            k,
            v,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.causal,
            g.window,
            None,
            kv_start,
            layout,
        )
        .expect("wgpu flash_attention_fwd");
    let out = read_f32(&out);
    let out = match layout {
        AttnOutLayout::HeadMajor => out,
        AttnOutLayout::TokenMajor => token_major_to_head_major(&out, g),
    };
    let lse = read_f32(&lse)
        .into_iter()
        .map(|x| if x <= -1e29 { f32::NEG_INFINITY } else { x })
        .collect();
    (out, lse)
}

fn check_case(g: KvStartGeom, starts: &[i32], label: &str) {
    let _lock = WGPU_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let Some((client, device)) = wgpu_setup() else {
        return;
    };
    let q_shape = [g.batch, g.num_heads, g.seq_q, g.head_dim];
    let kv_shape = [g.batch, g.num_kv_heads, g.seq_k, g.head_dim];
    let data = fixture_data(&g);
    let tensors = [
        Tensor::<WgpuRuntime>::from_slice(&data[0], &q_shape, &device).expect("q"),
        Tensor::<WgpuRuntime>::from_slice(&data[1], &kv_shape, &device).expect("k"),
        Tensor::<WgpuRuntime>::from_slice(&data[2], &kv_shape, &device).expect("v"),
    ];
    let kv_start = Tensor::<WgpuRuntime>::from_slice(starts, &[g.batch], &device).expect("starts");
    let (want_out, mut want_lse) =
        sliced_reference(&client, &tensors[0], &tensors[1], &tensors[2], &g, starts);
    for x in want_lse.iter_mut() {
        if *x <= -1e29 {
            *x = f32::NEG_INFINITY;
        }
    }

    for layout in [AttnOutLayout::HeadMajor, AttnOutLayout::TokenMajor] {
        let (got, lse) = run(&client, &tensors, &g, Some(&kv_start), layout);
        let tag = format!("{label} {layout:?}");
        let d_out = max_abs_diff(&got, &want_out, &format!("{tag} out"));
        let d_lse = max_abs_diff(&lse, &want_lse, &format!("{tag} lse"));
        assert_bits(
            &got,
            &want_out,
            &format!("{tag} out (max abs diff {d_out:.3e})"),
        );
        assert_bits(
            &lse,
            &want_lse,
            &format!("{tag} lse (max abs diff {d_lse:.3e})"),
        );
        for (i, e) in want_lse.iter().enumerate() {
            if e.is_infinite() {
                let row = &got[i * g.head_dim..(i + 1) * g.head_dim];
                assert!(
                    row.iter().all(|x| *x == 0.0),
                    "{tag}: dead row {i} is not zero"
                );
            }
        }
    }

    // Cross-backend: the CPU path on the same padded batch.
    let (cpu_client, cpu_device) = cpu_setup();
    let cpu = [
        Tensor::<CpuRuntime>::from_slice(&data[0], &q_shape, &cpu_device).expect("q"),
        Tensor::<CpuRuntime>::from_slice(&data[1], &kv_shape, &cpu_device).expect("k"),
        Tensor::<CpuRuntime>::from_slice(&data[2], &kv_shape, &cpu_device).expect("v"),
    ];
    let cpu_start =
        Tensor::<CpuRuntime>::from_slice(starts, &[g.batch], &cpu_device).expect("starts");
    let (cpu_out, _) = cpu_client
        .flash_attention_fwd(
            &cpu[0],
            &cpu[1],
            &cpu[2],
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.causal,
            g.window,
            None,
            Some(&cpu_start),
            AttnOutLayout::HeadMajor,
        )
        .expect("cpu flash_attention_fwd");
    let (got, _) = run(
        &client,
        &tensors,
        &g,
        Some(&kv_start),
        AttnOutLayout::HeadMajor,
    );
    let d_cpu = max_abs_diff(&got, &cpu_out.to_vec::<f32>(), &format!("{label} vs cpu"));
    assert!(d_cpu <= 1e-4, "{label}: wgpu vs cpu diff {d_cpu:.3e}");

    let zeros = Tensor::<WgpuRuntime>::zeros(&[g.batch], DType::I32, &device).expect("zeros");
    let none = run(&client, &tensors, &g, None, AttnOutLayout::HeadMajor);
    // The padding changes the answer, or the case would prove nothing.
    assert_ne!(
        none.0, got,
        "{label}: padded output equals the unpadded one"
    );
    let zero = run(
        &client,
        &tensors,
        &g,
        Some(&zeros),
        AttnOutLayout::HeadMajor,
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&none.0), bits(&zero.0), "{label}: zeros != None (out)");
    assert_eq!(bits(&none.1), bits(&zero.1), "{label}: zeros != None (lse)");
}

#[test]
fn decode() {
    check_case(geom(1, 40, 64, false, 0), &STARTS, "wgpu decode D=64 Sk=40");
}

#[test]
fn decode_dead_row() {
    check_case(
        geom(1, 40, 64, false, 0),
        &[0, 39, 40],
        "wgpu decode dead row",
    );
}

#[test]
fn prefill_causal() {
    check_case(
        geom(40, 40, 64, true, 0),
        &STARTS,
        "wgpu prefill causal S=40",
    );
}

#[test]
fn chunked_causal() {
    check_case(geom(8, 40, 64, true, 0), &STARTS, "wgpu chunked Sq=8 Sk=40");
}

#[test]
fn window_with_kv_start() {
    check_case(geom(40, 40, 64, true, 8), &STARTS, "wgpu window=8 S=40");
}

#[test]
fn short_query_non_causal() {
    check_case(
        geom(11, 40, 128, false, 0),
        &STARTS,
        "wgpu Sq=11 D=128 Sk=40",
    );
    check_case(
        geom(11, 11, 128, false, 0),
        &[0, 5, 11],
        "wgpu Sq=11 D=128 Sk=11 dead row",
    );
}

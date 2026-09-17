//! `AttnOutLayout::TokenMajor` parity for `FlashAttentionOps::flash_attention_fwd`.
//!
//! The contract: only the store address changes, so the token-major output is
//! BITWISE `permute([0, 2, 1, 3])` of the head-major output for the same
//! inputs on the same backend, and the LSE is identical. Every CUDA forward
//! tier is covered by geometry: the decode kernel (`S_q == 1`, both grid
//! shapes), the folded short-query path, the MQA/GQA tiled kernel, the general
//! `flash_v2.cu` kernel (head_dim 96, and a sliding window), and the
//! `kv_seq_len` override. The backward is covered through
//! `var_flash_attention`: the same loss through either layout yields
//! bitwise-equal dQ/dK/dV.

use super::helpers::*;
use boostr::nn::var_contiguous;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use boostr::ops::{AttnOutLayout, var_flash_attention};
use numr::autograd::{Var, backward, var_mul, var_permute, var_reshape, var_sum};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One forward geometry. `sk_cap` is the K/V capacity; `kv_seq_len` is the
/// override passed to the kernel (`None` uses the capacity).
#[derive(Clone, Copy)]
struct Geom {
    b: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    sq: usize,
    sk_cap: usize,
    kv_seq_len: Option<usize>,
    causal: bool,
    window: usize,
    label: &'static str,
}

#[allow(clippy::too_many_arguments)]
const fn g(
    b: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    sq: usize,
    sk: usize,
    causal: bool,
    window: usize,
    label: &'static str,
) -> Geom {
    Geom {
        b,
        h,
        h_kv,
        d,
        sq,
        sk_cap: sk,
        kv_seq_len: None,
        causal,
        window,
        label,
    }
}

/// Every CUDA forward tier by geometry (see the module doc).
const GEOMS: &[Geom] = &[
    g(2, 8, 2, 128, 1, 40, false, 0, "decode whole-sequence"),
    g(1, 2, 1, 64, 1, 8192, true, 0, "decode split-kv"),
    g(2, 8, 2, 64, 1, 300, false, 12, "decode windowed"),
    g(2, 16, 2, 128, 11, 11, false, 0, "folded short query gqa"),
    g(
        2,
        8,
        1,
        64,
        3,
        40,
        false,
        0,
        "folded short query mqa long kv",
    ),
    g(1, 4, 4, 32, 4, 4, false, 0, "folded short query mha d32"),
    g(2, 16, 2, 128, 40, 40, false, 0, "mqa_gqa tiled non-causal"),
    g(2, 16, 2, 128, 40, 40, true, 0, "mqa_gqa tiled causal"),
    g(
        1,
        4,
        4,
        64,
        37,
        37,
        true,
        0,
        "mqa_gqa tiled mha d64 ragged rows",
    ),
    g(2, 4, 2, 96, 20, 20, true, 0, "flash_v2 general d96"),
    g(1, 4, 4, 64, 32, 32, false, 8, "flash_v2 general windowed"),
    g(1, 2, 2, 192, 18, 18, false, 0, "flash_v2 general d192"),
    Geom {
        kv_seq_len: Some(40),
        ..g(
            2,
            8,
            2,
            128,
            5,
            64,
            false,
            0,
            "kv_seq_len override short query",
        )
    },
    Geom {
        kv_seq_len: Some(40),
        ..g(2, 8, 2, 128, 33, 64, true, 0, "kv_seq_len override tiled")
    },
];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn inputs(
    geom: &Geom,
    device: &numr::runtime::cpu::CpuDevice,
) -> (
    Tensor<numr::runtime::cpu::CpuRuntime>,
    Tensor<numr::runtime::cpu::CpuRuntime>,
    Tensor<numr::runtime::cpu::CpuRuntime>,
) {
    let q = det_tensor(&[geom.b, geom.h, geom.sq, geom.d], device);
    let k = det_tensor(&[geom.b, geom.h_kv, geom.sk_cap, geom.d], device);
    let v = det_tensor(&[geom.b, geom.h_kv, geom.sk_cap, geom.d], device);
    (q, k, v)
}

/// Runs both layouts on one backend and asserts the token-major output is
/// bitwise the permuted head-major output, with identical LSE. Returns the
/// token-major output as host floats.
fn assert_layouts_agree<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    geom: &Geom,
    backend: &str,
) -> Vec<f32>
where
    R: Runtime<DType = DType>,
    C: FlashAttentionOps<R>,
{
    let run = |layout| {
        client
            .flash_attention_fwd(
                q,
                k,
                v,
                geom.h,
                geom.h_kv,
                geom.d,
                geom.causal,
                geom.window,
                geom.kv_seq_len,
                layout,
            )
            .unwrap_or_else(|e| panic!("{backend} {}: {layout:?} failed: {e}", geom.label))
    };
    let (out_hm, lse_hm) = run(AttnOutLayout::HeadMajor);
    let (out_tm, lse_tm) = run(AttnOutLayout::TokenMajor);

    assert_eq!(
        out_hm.shape(),
        &[geom.b, geom.h, geom.sq, geom.d],
        "{backend} {}: head-major shape",
        geom.label
    );
    assert_eq!(
        out_tm.shape(),
        &[geom.b, geom.sq, geom.h, geom.d],
        "{backend} {}: token-major shape",
        geom.label
    );
    assert!(
        out_tm.is_contiguous(),
        "{backend} {}: token-major output must be dense",
        geom.label
    );
    assert_eq!(
        lse_tm.shape(),
        &[geom.b, geom.h, geom.sq],
        "{backend} {}: lse shape is layout-independent",
        geom.label
    );

    let permuted = out_hm
        .permute(&[0, 2, 1, 3])
        .expect("permute")
        .contiguous()
        .expect("contiguous")
        .to_vec::<f32>();
    let tm = out_tm.to_vec::<f32>();
    assert_eq!(
        bits(&tm),
        bits(&permuted),
        "{backend} {}: token-major is not bitwise permute(head-major)",
        geom.label
    );
    assert_eq!(
        bits(&lse_tm.to_vec::<f32>()),
        bits(&lse_hm.to_vec::<f32>()),
        "{backend} {}: lse differs between layouts",
        geom.label
    );
    tm
}

/// CPU: the composed path re-lays its matmul result; the decode fast path
/// reshapes. Both must be bitwise the permuted head-major result.
#[test]
fn token_major_is_permuted_head_major_cpu() {
    let (client, device) = setup_cpu();
    for geom in GEOMS {
        let (q, k, v) = inputs(geom, &device);
        assert_layouts_agree(&client, &q, &k, &v, geom, "cpu");
    }
}

/// CUDA: every forward tier, bitwise against its own head-major store, and
/// the token-major result against CPU token-major at the usual parity
/// tolerance.
#[cfg(feature = "cuda")]
#[test]
fn token_major_is_permuted_head_major_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    with_cuda_backend(|cuda_client, cuda_device| {
        for geom in GEOMS {
            let (q, k, v) = inputs(geom, &cpu_device);
            let cpu_tm = assert_layouts_agree(&cpu_client, &q, &k, &v, geom, "cpu");
            let up = |t: &Tensor<numr::runtime::cpu::CpuRuntime>| {
                Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &cuda_device).expect("upload")
            };
            let (q_c, k_c, v_c) = (up(&q), up(&k), up(&v));
            let cuda_tm = assert_layouts_agree(&cuda_client, &q_c, &k_c, &v_c, geom, "cuda");
            assert_parity_f32_tol(
                &cuda_tm,
                &cpu_tm,
                &format!("token-major CUDA vs CPU: {}", geom.label),
                1e-4,
                1e-6,
            );
        }
    });
}

/// CUDA at BF16: the store flag must hold at the half-precision entry points
/// too, where the kernel rounds once at the store.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn token_major_is_permuted_head_major_cuda_bf16() {
    let (_cpu_client, cpu_device) = setup_cpu();
    with_cuda_backend(|cuda_client, cuda_device| {
        for geom in GEOMS {
            let (q, k, v) = inputs(geom, &cpu_device);
            let up = |t: &Tensor<numr::runtime::cpu::CpuRuntime>| {
                Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &cuda_device)
                    .expect("upload")
                    .to_dtype(DType::BF16)
                    .expect("cast to bf16")
            };
            let (q_c, k_c, v_c) = (up(&q), up(&k), up(&v));
            let run = |layout| {
                cuda_client
                    .flash_attention_fwd(
                        &q_c,
                        &k_c,
                        &v_c,
                        geom.h,
                        geom.h_kv,
                        geom.d,
                        geom.causal,
                        geom.window,
                        geom.kv_seq_len,
                        layout,
                    )
                    .unwrap_or_else(|e| panic!("cuda bf16 {}: {layout:?} failed: {e}", geom.label))
                    .0
                    .to_dtype(DType::F32)
                    .expect("cast to f32")
            };
            let hm = run(AttnOutLayout::HeadMajor);
            let tm = run(AttnOutLayout::TokenMajor);
            assert_eq!(tm.shape(), &[geom.b, geom.sq, geom.h, geom.d]);
            let permuted = hm
                .permute(&[0, 2, 1, 3])
                .expect("permute")
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            assert_eq!(
                bits(&tm.to_vec::<f32>()),
                bits(&permuted),
                "cuda bf16 {}: token-major is not bitwise permute(head-major)",
                geom.label
            );
        }
    });
}

/// WGPU: the shader's store address flag, bitwise against its own head-major
/// store and against CPU at parity tolerance.
#[cfg(feature = "wgpu")]
#[test]
fn token_major_is_permuted_head_major_wgpu() {
    let (cpu_client, cpu_device) = setup_cpu();
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for geom in GEOMS {
            let (q, k, v) = inputs(geom, &cpu_device);
            let cpu_tm = assert_layouts_agree(&cpu_client, &q, &k, &v, geom, "cpu");
            let up = |t: &Tensor<numr::runtime::cpu::CpuRuntime>| {
                Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &wgpu_device).expect("upload")
            };
            let (q_w, k_w, v_w) = (up(&q), up(&k), up(&v));
            let wgpu_tm = assert_layouts_agree(&wgpu_client, &q_w, &k_w, &v_w, geom, "wgpu");
            assert_parity_f32_tol(
                &wgpu_tm,
                &cpu_tm,
                &format!("token-major WGPU vs CPU: {}", geom.label),
                1e-4,
                1e-6,
            );
        }
    });
}

/// Forward + backward through `var_flash_attention` in one layout, ending in
/// the `[B, S_q, H * D]` projection input either way. Returns `(dq, dk, dv)`
/// as host floats.
fn grads_through_layout<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    weight: &Tensor<R>,
    geom: &Geom,
    layout: AttnOutLayout,
) -> [Vec<f32>; 3]
where
    R: Runtime<DType = DType>,
    R::Client: FlashAttentionOps<R> + numr::ops::TensorOps<R>,
    C: numr::runtime::RuntimeClient<R> + numr::ops::TensorOps<R>,
{
    let q = Var::new(q.clone(), true);
    let k = Var::new(k.clone(), true);
    let v = Var::new(v.clone(), true);
    let out = var_flash_attention(
        &q,
        &k,
        &v,
        geom.h,
        geom.h_kv,
        geom.d,
        geom.causal,
        geom.window,
        layout,
    )
    .unwrap_or_else(|e| panic!("{}: {layout:?} forward failed: {e}", geom.label));
    // Both layouts end as the dense `[B, S_q, H, D]` the projection reads.
    let token_major = match layout {
        AttnOutLayout::HeadMajor => {
            let p = var_permute(&out, &[0, 2, 1, 3]).expect("permute");
            var_contiguous(&p).expect("contiguous")
        }
        AttnOutLayout::TokenMajor => out,
    };
    let flat = var_reshape(&token_major, &[geom.b, geom.sq, geom.h * geom.d]).expect("reshape");
    // A non-uniform upstream gradient, so a wrong store address changes dQ.
    let weighted = var_mul(&flat, &Var::new(weight.clone(), false), client).expect("mul");
    let loss = var_sum(&weighted, &[0, 1, 2], false, client).expect("sum");
    let grads = backward(&loss, client)
        .unwrap_or_else(|e| panic!("{}: {layout:?} backward failed: {e}", geom.label));
    let take = |var: &Var<R>, name: &str| {
        grads
            .get(var.tensor().id())
            .unwrap_or_else(|| panic!("{}: {layout:?} has no {name}", geom.label))
            .contiguous()
            .expect("contiguous grad")
            .to_vec::<f32>()
    };
    [take(&q, "dQ"), take(&k, "dK"), take(&v, "dV")]
}

/// Geometries with `S_k == S_q`, which the fused backward kernels require.
fn backward_geoms() -> Vec<Geom> {
    GEOMS
        .iter()
        .copied()
        .filter(|g| g.sq == g.sk_cap && g.kv_seq_len.is_none() && g.sq > 1)
        .collect()
}

fn assert_grads_bitwise(a: &[Vec<f32>; 3], b: &[Vec<f32>; 3], label: &str) {
    for (name, (x, y)) in ["dQ", "dK", "dV"].iter().zip(a.iter().zip(b.iter())) {
        assert_eq!(
            bits(x),
            bits(y),
            "{label}: {name} differs between token-major and head-major"
        );
    }
}

/// CPU: the backward node permutes a token-major `dO`/`O` back before the
/// composed backward, so the gradients are bitwise those of the head-major
/// graph, whose `PermuteBackward` produces the same `dO`.
#[test]
fn token_major_backward_matches_head_major_cpu() {
    let (client, device) = setup_cpu();
    for geom in backward_geoms() {
        let (q, k, v) = inputs(&geom, &device);
        let w = det_tensor(&[geom.b, geom.sq, geom.h * geom.d], &device);
        let hm = grads_through_layout(&client, &q, &k, &v, &w, &geom, AttnOutLayout::HeadMajor);
        let tm = grads_through_layout(&client, &q, &k, &v, &w, &geom, AttnOutLayout::TokenMajor);
        assert_grads_bitwise(&tm, &hm, &format!("cpu {}", geom.label));
    }
}

/// CUDA: same statement through the fused backward kernels. The permuted
/// `dO` reaches the kernel as the same bytes either way; dQ, dK and dV all
/// accumulate through `atomicAdd`, whose arrival order is not fixed across
/// launches, so this holds the two runs to the tolerance every CUDA backward
/// parity test uses. The bitwise statement is the CPU test above.
#[cfg(feature = "cuda")]
#[test]
fn token_major_backward_matches_head_major_cuda() {
    let (_cpu_client, cpu_device) = setup_cpu();
    with_cuda_backend(|cuda_client, cuda_device| {
        for geom in backward_geoms() {
            let (q, k, v) = inputs(&geom, &cpu_device);
            let w = det_tensor(&[geom.b, geom.sq, geom.h * geom.d], &cpu_device);
            let up = |t: &Tensor<numr::runtime::cpu::CpuRuntime>| {
                Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &cuda_device).expect("upload")
            };
            let (q_c, k_c, v_c, w_c) = (up(&q), up(&k), up(&v), up(&w));
            let hm = grads_through_layout(
                &cuda_client,
                &q_c,
                &k_c,
                &v_c,
                &w_c,
                &geom,
                AttnOutLayout::HeadMajor,
            );
            let tm = grads_through_layout(
                &cuda_client,
                &q_c,
                &k_c,
                &v_c,
                &w_c,
                &geom,
                AttnOutLayout::TokenMajor,
            );
            for (name, i) in [("dQ", 0usize), ("dK", 1), ("dV", 2)] {
                assert_parity_f32_relaxed(
                    &tm[i],
                    &hm[i],
                    &format!("cuda {}: {name} token-major vs head-major", geom.label),
                );
            }
        }
    });
}

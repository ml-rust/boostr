//! Backend parity for `var_varlen_attention`: output, dQ, dK and dV on CUDA
//! and WebGPU against the CPU run on the same packed inputs.
//!
//! WebGPU varlen runs MHA in F32 only, so the WebGPU case uses MHA.
//! CUDA also runs a GQA case.

use super::helpers::*;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use boostr::ops::{PackedSegments, var_varlen_attention};
use numr::autograd::{Var, backward, var_mul, var_sum};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One packed geometry: segment offsets, heads, kv heads, head dim, causal.
#[derive(Clone)]
struct Geom {
    offsets: Vec<i32>,
    h: usize,
    h_kv: usize,
    d: usize,
    causal: bool,
    label: &'static str,
}

impl Geom {
    fn tokens(&self) -> usize {
        *self.offsets.last().unwrap_or(&0) as usize
    }
}

fn host(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.21 + phase).sin() * 0.5)
        .collect()
}

/// Forward plus backward on one backend. Returns `[out, dq, dk, dv]` as host floats.
fn run<R, C>(client: &C, device: &R::Device, g: &Geom) -> [Vec<f32>; 4]
where
    R: Runtime<DType = DType>,
    R::Client: VarLenAttentionOps<R> + ScalarOps<R> + TensorOps<R>,
    C: RuntimeClient<R> + VarLenAttentionOps<R> + ScalarOps<R> + TensorOps<R>,
{
    let t = g.tokens();
    let (nq, nkv) = (t * g.h * g.d, t * g.h_kv * g.d);
    let make = |data: Vec<f32>, shape: &[usize], grad: bool| {
        let tensor = Tensor::<R>::from_slice(&data, shape, device).expect("upload input");
        Var::new(tensor, grad)
    };
    let q = make(host(nq, 0.0), &[t, g.h, g.d], true);
    let k = make(host(nkv, 1.3), &[t, g.h_kv, g.d], true);
    let v = make(host(nkv, 2.9), &[t, g.h_kv, g.d], true);
    let w = make(host(nq, 4.1), &[t, g.h, g.d], false);
    let segs = PackedSegments::<R>::from_offsets(&g.offsets, t, device).expect("segments");
    let scale = 1.0 / (g.d as f64).sqrt();

    let out = var_varlen_attention(client, &q, &k, &v, &segs, scale, g.causal)
        .unwrap_or_else(|e| panic!("{}: forward failed: {e}", g.label));
    let weighted = var_mul(&out, &w, client).expect("mul");
    let loss = var_sum(&weighted, &[0, 1, 2], false, client).expect("sum");
    let grads = backward(&loss, client).unwrap_or_else(|e| panic!("{}: backward: {e}", g.label));
    let take = |var: &Var<R>, name: &str| {
        grads
            .get(var.tensor().id())
            .unwrap_or_else(|| panic!("{}: no {name}", g.label))
            .contiguous()
            .expect("contiguous grad")
            .to_vec::<f32>()
    };
    let out_host = out
        .tensor()
        .contiguous()
        .expect("contiguous out")
        .to_vec::<f32>();
    [out_host, take(&q, "dQ"), take(&k, "dK"), take(&v, "dV")]
}

fn assert_all(got: &[Vec<f32>; 4], want: &[Vec<f32>; 4], label: &str, backend: &str) {
    for (name, (a, b)) in ["out", "dQ", "dK", "dV"].iter().zip(got.iter().zip(want)) {
        assert_parity_f32_relaxed(a, b, &format!("{label} {name} {backend} vs CPU"));
    }
}

fn mha_geom() -> Geom {
    Geom {
        offsets: vec![0, 5, 12, 16],
        h: 2,
        h_kv: 2,
        d: 64,
        causal: true,
        label: "var_varlen mha causal",
    }
}

fn gqa_geom() -> Geom {
    Geom {
        offsets: vec![0, 7, 10],
        h: 4,
        h_kv: 2,
        d: 64,
        causal: true,
        label: "var_varlen gqa causal",
    }
}

#[test]
fn test_var_varlen_attention_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let mha = mha_geom();
    let gqa = gqa_geom();
    let cpu_mha = run::<numr::runtime::cpu::CpuRuntime, _>(&cpu_client, &cpu_device, &mha);
    let cpu_gqa = run::<numr::runtime::cpu::CpuRuntime, _>(&cpu_client, &cpu_device, &gqa);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let got = run::<numr::runtime::cuda::CudaRuntime, _>(&cuda_client, &cuda_device, &mha);
        assert_all(&got, &cpu_mha, mha.label, "CUDA");
        let got = run::<numr::runtime::cuda::CudaRuntime, _>(&cuda_client, &cuda_device, &gqa);
        assert_all(&got, &cpu_gqa, gqa.label, "CUDA");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let got = run::<numr::runtime::wgpu::WgpuRuntime, _>(&wgpu_client, &wgpu_device, &mha);
        assert_all(&got, &cpu_mha, mha.label, "WGPU");
    });

    // The CPU runs are the reference. Check they are finite so a NaN
    // reference cannot pass every comparison above.
    for v in cpu_mha.iter().chain(cpu_gqa.iter()) {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "CPU reference holds a non-finite value"
        );
    }
}

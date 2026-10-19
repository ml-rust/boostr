//! Backend parity for `SpeculativeOps::speculative_verify`.
//!
//! Every probability is a multiple of 2^-16, so every running sum is exact in F32.
//! The uniforms are inputs. So each backend must return the CPU's `(m, next)` exactly.

use super::helpers::*;
use boostr::ops::traits::inference::speculative::SpeculativeOps;
use numr::dtype::DType;
use numr::ops::AdvancedRandomOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// One verify call on host data.
struct Case {
    name: &'static str,
    /// `[B, K, V]`.
    bkv: [usize; 3],
    p: Vec<f32>,
    q: Option<Vec<f32>>,
    d: Vec<i32>,
    u: Vec<f32>,
}

const UNIT: f32 = 1.0 / 65536.0;

/// The draft id of run `r` at position `i`.
fn draft_id(r: usize, i: usize, v: usize, salt: usize) -> i32 {
    ((r * 31 + i * 17 + salt) % v) as i32
}

/// `rows` dyadic rows of `v` entries. Even runs get a spike of 1/2 on the draft id.
fn rows(b: usize, n: usize, k: usize, v: usize, salt: usize, spike: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n * v];
    for r in 0..b {
        for i in 0..n {
            let base = (r * n + i) * v;
            for j in 0..v {
                out[base + j] = ((j * 7 + i * 3 + r * 5 + salt) % 9) as f32 * UNIT;
            }
            if i < k && r % 2 == 0 {
                out[base + draft_id(r, i, v, 0) as usize] = spike;
            }
        }
    }
    out
}

fn case(name: &'static str, [b, k, v]: [usize; 3], general: bool, key: u64) -> Case {
    let (client, _) = setup_cpu();
    let u = client
        .philox_uniform(&[b, k + 1], key, 0, DType::F32)
        .expect("uniforms")
        .to_vec::<f32>();
    let d = (0..b * k)
        .map(|x| draft_id(x / k.max(1), x % k.max(1), v, 0))
        .collect();
    Case {
        name,
        bkv: [b, k, v],
        p: rows(b, k + 1, k, v, 1, 0.5),
        q: general.then(|| rows(b, k, k, v, 4, 0.25)),
        d,
        u,
    }
}

fn cases() -> Vec<Case> {
    let mut out_of_range = case("out of range drafts", [4, 3, 32], false, 6);
    out_of_range.d[1] = 35;
    out_of_range.d[3] = -1;
    out_of_range.d[7] = i32::MAX;
    vec![
        case("one-hot", [8, 4, 32], false, 1),
        case("general", [8, 4, 32], true, 2),
        case("one-hot wide vocab", [4, 3, 5003], false, 3),
        case("general wide vocab", [4, 3, 5003], true, 4),
        case("zero drafts", [4, 0, 5003], false, 5),
        out_of_range,
    ]
}

fn run<R: Runtime<DType = numr::dtype::DType>, C: SpeculativeOps<R>>(
    client: &C,
    device: &R::Device,
    c: &Case,
) -> Vec<i32> {
    let [b, k, v] = c.bkv;
    let p = Tensor::<R>::from_slice(&c.p, &[b, k + 1, v], device).expect("target_probs");
    let q =
        c.q.as_ref()
            .map(|q| Tensor::<R>::from_slice(q, &[b, k, v], device).expect("draft_probs"));
    let d = Tensor::<R>::from_slice(&c.d, &[b, k], device).expect("draft_tokens");
    let u = Tensor::<R>::from_slice(&c.u, &[b, k + 1], device).expect("uniforms");
    client
        .speculative_verify(&p, q.as_ref(), &d, &u)
        .unwrap_or_else(|e| panic!("{}: {e}", c.name))
        .to_vec::<i32>()
}

#[test]
fn test_speculative_verify_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let cases = cases();
    let reference: Vec<Vec<i32>> = cases
        .iter()
        .map(|c| run::<CpuRuntime, _>(&cpu_client, &cpu_device, c))
        .collect();

    let (mut any_accept, mut any_reject) = (false, false);
    for (c, out) in cases.iter().zip(&reference) {
        let [b, k, v] = c.bkv;
        assert_eq!(out.len(), b * 2, "{}: result shape", c.name);
        for r in 0..b {
            let (m, next) = (out[2 * r] as usize, out[2 * r + 1] as usize);
            assert!(m <= k, "{} row {r}: m {m} > K {k}", c.name);
            assert!(next < v, "{} row {r}: next {next} >= V {v}", c.name);
            assert!(
                c.p[(r * (k + 1) + m) * v + next] > 0.0,
                "{} row {r}: next {next} has no target mass",
                c.name
            );
            any_accept |= m > 0;
            any_reject |= m < k;
        }
    }
    assert!(
        any_accept && any_reject,
        "the cases must cover accepts and rejects"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        for (c, want) in cases.iter().zip(&reference) {
            let got = run::<numr::runtime::cuda::CudaRuntime, _>(&client, &device, c);
            assert_eq!(&got, want, "CUDA vs CPU: {}", c.name);
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        for (c, want) in cases.iter().zip(&reference) {
            let got = run::<numr::runtime::wgpu::WgpuRuntime, _>(&client, &device, c);
            assert_eq!(&got, want, "WGPU vs CPU: {}", c.name);
        }
    });
}

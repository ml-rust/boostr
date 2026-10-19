//! Integration tests for `var_varlen_attention`, the autograd op over packed
//! `[T, H, D]` attention split by `PackedSegments`.
//!
//! Reference: `var_attention_with_bias` on the same tokens laid out as one
//! `[1, H, T, D]` sequence, with a block-diagonal bias that blocks every
//! cross-segment pair (and future keys when causal). Its backward comes from
//! numr's autograd graph, so it is independent of the varlen kernels.
//!
//! Tolerance for F32: both sides compute the same sums in a different order
//! (scalar loops against matmul + softmax). The rounding gap grows with
//! `sqrt(D + T) * eps * |value|`, about 1e-6 here. `ATOL = 1e-5` and
//! `RTOL = 1e-4` leave a 10x margin and still catch any masking error.

use boostr::ops::{
    AttentionCausality, PackedSegments, var_attention_with_bias, var_varlen_attention,
};
use numr::autograd::{GradStore, Var, backward, var_mul, var_sum};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuClient::new(device.clone()), device)
}

fn det(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 0.5)
        .collect()
}

fn leaf(data: &[f32], shape: &[usize], dev: &CpuDevice) -> TestResult<Var<CpuRuntime>> {
    Ok(Var::new(
        Tensor::<CpuRuntime>::from_slice(data, shape, dev)?,
        true,
    ))
}

fn grad(grads: &GradStore<CpuRuntime>, v: &Var<CpuRuntime>, name: &str) -> TestResult<Vec<f32>> {
    let g = grads
        .get(v.tensor().id())
        .ok_or_else(|| format!("no gradient for {name}"))?;
    Ok(g.contiguous()?.to_vec::<f32>())
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = ATOL + RTOL * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w}, diff {} > tol {tol}",
            (g - w).abs()
        );
    }
}

/// `[T, H, D]` packed data to `[H, T, D]` dense data, repeating each kv head `rep` times.
fn to_dense(packed: &[f32], t: usize, h: usize, d: usize, rep: usize) -> Vec<f32> {
    let mut out = vec![0.0; h * rep * t * d];
    for tok in 0..t {
        for hh in 0..h * rep {
            let src = (tok * h + hh / rep) * d;
            let dst = (hh * t + tok) * d;
            out[dst..dst + d].copy_from_slice(&packed[src..src + d]);
        }
    }
    out
}

/// `[H * rep, T, D]` dense gradient to `[T, H, D]` packed, summing each group of `rep` heads.
fn to_packed(dense: &[f32], t: usize, h: usize, d: usize, rep: usize) -> Vec<f32> {
    let mut out = vec![0.0; t * h * d];
    for hh in 0..h * rep {
        for tok in 0..t {
            let src = (hh * t + tok) * d;
            let dst = (tok * h + hh / rep) * d;
            for x in 0..d {
                out[dst + x] += dense[src + x];
            }
        }
    }
    out
}

/// `[1, 1, T, T]` bias: 0 where the query sees the key, `f32::MIN` elsewhere.
fn block_bias(offsets: &[i32], causal: bool, dev: &CpuDevice) -> TestResult<Tensor<CpuRuntime>> {
    let t = *offsets.last().ok_or("empty offsets")? as usize;
    let seg = |tok: usize| {
        offsets
            .windows(2)
            .position(|w| (w[0] as usize..w[1] as usize).contains(&tok))
    };
    let mut bias = vec![f32::MIN; t * t];
    for i in 0..t {
        for j in 0..t {
            if seg(i) == seg(j) && (!causal || j <= i) {
                bias[i * t + j] = 0.0;
            }
        }
    }
    Ok(Tensor::<CpuRuntime>::from_slice(&bias, &[1, 1, t, t], dev)?)
}

struct Grads {
    out: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

struct Case {
    offsets: Vec<i32>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
}

impl Case {
    fn tokens(&self) -> usize {
        *self.offsets.last().unwrap_or(&0) as usize
    }

    fn inputs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let t = self.tokens();
        let nq = t * self.heads * self.head_dim;
        let nkv = t * self.kv_heads * self.head_dim;
        (det(nq, 0.0), det(nkv, 1.1), det(nkv, 2.3), det(nq, 4.7))
    }

    /// Varlen op: weighted-sum loss so dOut is not uniform.
    fn run_varlen(&self, q: &[f32], k: &[f32], v: &[f32], w: &[f32]) -> TestResult<Grads> {
        let (client, dev) = setup();
        let (t, h, hkv, d) = (self.tokens(), self.heads, self.kv_heads, self.head_dim);
        let qv = leaf(q, &[t, h, d], &dev)?;
        let kv = leaf(k, &[t, hkv, d], &dev)?;
        let vv = leaf(v, &[t, hkv, d], &dev)?;
        let wv = Var::new(
            Tensor::<CpuRuntime>::from_slice(w, &[t, h, d], &dev)?,
            false,
        );
        let segs = PackedSegments::<CpuRuntime>::from_offsets(&self.offsets, t, &dev)?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = var_varlen_attention(&client, &qv, &kv, &vv, &segs, scale, self.causal)?;
        let loss = var_sum(&var_mul(&out, &wv, &client)?, &[0, 1, 2], false, &client)?;
        let grads = backward(&loss, &client)?;
        Ok(Grads {
            out: out.tensor().contiguous()?.to_vec::<f32>(),
            dq: grad(&grads, &qv, "q")?,
            dk: grad(&grads, &kv, "k")?,
            dv: grad(&grads, &vv, "v")?,
        })
    }

    /// Dense reference with a block-diagonal bias, results mapped back to packed layout.
    fn run_reference(&self, q: &[f32], k: &[f32], v: &[f32], w: &[f32]) -> TestResult<Grads> {
        let (client, dev) = setup();
        let (t, h, hkv, d) = (self.tokens(), self.heads, self.kv_heads, self.head_dim);
        let rep = h / hkv;
        let qv = leaf(&to_dense(q, t, h, d, 1), &[1, h, t, d], &dev)?;
        let kv = leaf(&to_dense(k, t, hkv, d, rep), &[1, h, t, d], &dev)?;
        let vv = leaf(&to_dense(v, t, hkv, d, rep), &[1, h, t, d], &dev)?;
        let wd = to_dense(w, t, h, d, 1);
        let wv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&wd, &[1, h, t, d], &dev)?,
            false,
        );
        let bias = block_bias(&self.offsets, self.causal, &dev)?;
        let out =
            var_attention_with_bias(&client, &qv, &kv, &vv, &bias, AttentionCausality::InBias, h)?;
        let loss = var_sum(&var_mul(&out, &wv, &client)?, &[0, 1, 2, 3], false, &client)?;
        let grads = backward(&loss, &client)?;
        let out_dense = out.tensor().contiguous()?.to_vec::<f32>();
        Ok(Grads {
            out: to_packed(&out_dense, t, h, d, 1),
            dq: to_packed(&grad(&grads, &qv, "q_ref")?, t, h, d, 1),
            dk: to_packed(&grad(&grads, &kv, "k_ref")?, t, hkv, d, rep),
            dv: to_packed(&grad(&grads, &vv, "v_ref")?, t, hkv, d, rep),
        })
    }

    fn check_against_reference(&self, label: &str) -> TestResult {
        let (q, k, v, w) = self.inputs();
        let got = self.run_varlen(&q, &k, &v, &w)?;
        let want = self.run_reference(&q, &k, &v, &w)?;
        assert_close(&got.out, &want.out, &format!("{label} out"));
        assert_close(&got.dq, &want.dq, &format!("{label} dQ"));
        assert_close(&got.dk, &want.dk, &format!("{label} dK"));
        assert_close(&got.dv, &want.dv, &format!("{label} dV"));
        Ok(())
    }
}

#[test]
fn causal_mha_matches_block_diagonal_reference() -> TestResult {
    Case {
        offsets: vec![0, 3, 8, 12],
        heads: 2,
        kv_heads: 2,
        head_dim: 8,
        causal: true,
    }
    .check_against_reference("causal mha")
}

#[test]
fn causal_gqa_matches_block_diagonal_reference() -> TestResult {
    Case {
        offsets: vec![0, 5, 7, 13],
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        causal: true,
    }
    .check_against_reference("causal gqa")
}

#[test]
fn non_causal_matches_block_diagonal_reference() -> TestResult {
    Case {
        offsets: vec![0, 4, 10],
        heads: 4,
        kv_heads: 1,
        head_dim: 8,
        causal: false,
    }
    .check_against_reference("non-causal mqa")
}

/// One segment with causal masking is plain causal attention: the reference
/// adds its own causal mask to a zero bias.
#[test]
fn single_segment_equals_plain_causal_attention() -> TestResult {
    let case = Case {
        offsets: vec![0, 9],
        heads: 2,
        kv_heads: 2,
        head_dim: 8,
        causal: true,
    };
    let (q, k, v, w) = case.inputs();
    let got = case.run_varlen(&q, &k, &v, &w)?;

    let (client, dev) = setup();
    let (t, h, d) = (9, 2, 8);
    let qv = leaf(&to_dense(&q, t, h, d, 1), &[1, h, t, d], &dev)?;
    let kv = leaf(&to_dense(&k, t, h, d, 1), &[1, h, t, d], &dev)?;
    let vv = leaf(&to_dense(&v, t, h, d, 1), &[1, h, t, d], &dev)?;
    let zero = Tensor::<CpuRuntime>::zeros(&[1, 1, t, t], DType::F32, &dev)?;
    let causal = AttentionCausality::Mask { window_size: 0 };
    let out = var_attention_with_bias(&client, &qv, &kv, &vv, &zero, causal, h)?;
    let want = to_packed(&out.tensor().contiguous()?.to_vec::<f32>(), t, h, d, 1);
    assert_close(&got.out, &want, "single segment out");
    Ok(())
}

/// Changing segment A's q, k and v must leave segment B's output and dQ
/// bitwise unchanged. Any cross-segment read shows up as a changed bit.
#[test]
fn perturbing_one_segment_leaves_the_other_bitwise_unchanged() -> TestResult {
    for causal in [true, false] {
        let case = Case {
            offsets: vec![0, 4, 11],
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            causal,
        };
        let (q, k, v, w) = case.inputs();
        let base = case.run_varlen(&q, &k, &v, &w)?;

        let (q_row, kv_row) = (case.heads * case.head_dim, case.kv_heads * case.head_dim);
        let split = 4;
        let bump = |data: &[f32], row: usize| -> Vec<f32> {
            data.iter()
                .enumerate()
                .map(|(i, &x)| if i < split * row { x * -3.0 + 0.25 } else { x })
                .collect()
        };
        let moved = case.run_varlen(&bump(&q, q_row), &bump(&k, kv_row), &bump(&v, kv_row), &w)?;

        assert_eq!(
            base.out[split * q_row..],
            moved.out[split * q_row..],
            "causal={causal}: out B"
        );
        assert_eq!(
            base.dq[split * q_row..],
            moved.dq[split * q_row..],
            "causal={causal}: dQ B"
        );
        assert_eq!(
            base.dk[split * kv_row..],
            moved.dk[split * kv_row..],
            "causal={causal}: dK B"
        );
        assert_eq!(
            base.dv[split * kv_row..],
            moved.dv[split * kv_row..],
            "causal={causal}: dV B"
        );
        assert_ne!(
            base.out[..split * q_row],
            moved.out[..split * q_row],
            "causal={causal}: out A"
        );
    }
    Ok(())
}

#[test]
fn unsupported_dtype_is_refused() -> TestResult {
    let (client, dev) = setup();
    let q = Var::new(
        Tensor::<CpuRuntime>::zeros(&[4, 2, 8], DType::F64, &dev)?,
        true,
    );
    let segs = PackedSegments::<CpuRuntime>::from_lengths(&[4], &dev)?;
    let err = var_varlen_attention(&client, &q, &q, &q, &segs, 0.25, true)
        .err()
        .ok_or("F64 input was accepted")?;
    assert!(
        err.to_string().contains("supports F32, F16 and BF16"),
        "{err}"
    );
    Ok(())
}

/// Half-precision inputs run through the F32 kernel and come back in their dtype.
///
/// The reference is the F32 run on the same values rounded to the half type,
/// so only the final rounding of out, dQ, dK and dV separates the two.
/// That rounding is at most one half-ulp: 2^-11 relative for F16, 2^-8 for BF16.
#[cfg(feature = "f16")]
mod half_precision {
    use super::*;
    use numr::ops::TypeConversionOps;

    fn run(dtype: DType, rtol: f32) -> TestResult {
        let (client, dev) = setup();
        let case = Case {
            offsets: vec![0, 3, 9],
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            causal: true,
        };
        let (t, h, hkv, d) = (case.tokens(), case.heads, case.kv_heads, case.head_dim);
        let (q, k, v, _) = case.inputs();
        let half = |data: &[f32], shape: &[usize]| -> TestResult<Tensor<CpuRuntime>> {
            Ok(client.cast(&Tensor::<CpuRuntime>::from_slice(data, shape, &dev)?, dtype)?)
        };
        let qh = Var::new(half(&q, &[t, h, d])?, true);
        let kh = Var::new(half(&k, &[t, hkv, d])?, true);
        let vh = Var::new(half(&v, &[t, hkv, d])?, true);
        let segs = PackedSegments::<CpuRuntime>::from_offsets(&case.offsets, t, &dev)?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = var_varlen_attention(&client, &qh, &kh, &vh, &segs, scale, true)?;
        assert_eq!(out.tensor().dtype(), dtype);
        let loss = var_sum(&out, &[0, 1, 2], false, &client)?;
        let grads = backward(&loss, &client)?;

        let to_f32 = |t: &Tensor<CpuRuntime>| -> TestResult<Vec<f32>> {
            assert_eq!(t.dtype(), dtype);
            Ok(client.cast(&t.contiguous()?, DType::F32)?.to_vec::<f32>())
        };
        let get = |v: &Var<CpuRuntime>| -> TestResult<Vec<f32>> {
            to_f32(grads.get(v.tensor().id()).ok_or("missing gradient")?)
        };
        let rounded = |v: &Var<CpuRuntime>| -> TestResult<Vec<f32>> { to_f32(v.tensor()) };
        let ones = vec![1.0f32; t * h * d];
        let want = case.run_varlen(&rounded(&qh)?, &rounded(&kh)?, &rounded(&vh)?, &ones)?;

        let pairs = [
            (to_f32(out.tensor())?, want.out, "out"),
            (get(&qh)?, want.dq, "dQ"),
            (get(&kh)?, want.dk, "dK"),
            (get(&vh)?, want.dv, "dV"),
        ];
        for (got, exp, name) in pairs {
            // A gradient entry sums larger terms that cancel, so its rounding
            // error follows the tensor's largest magnitude, not the entry.
            let scale = exp.iter().fold(0.0f32, |m, e| m.max(e.abs()));
            for (i, (&g, &e)) in got.iter().zip(&exp).enumerate() {
                let tol = 1e-6 + rtol * e.abs() + 4.0 * rtol * scale;
                assert!((g - e).abs() <= tol, "{dtype:?} {name}[{i}]: {g} vs {e}");
            }
        }
        Ok(())
    }

    #[test]
    fn f16_matches_f32_on_rounded_inputs() -> TestResult {
        run(DType::F16, 2.0f32.powi(-11))
    }

    #[test]
    fn bf16_matches_f32_on_rounded_inputs() -> TestResult {
        run(DType::BF16, 2.0f32.powi(-8))
    }
}

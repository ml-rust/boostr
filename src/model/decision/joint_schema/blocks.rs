//! The building blocks of a joint-schema head, inference only and in one
//! compute dtype: [`Affine`] (a linear map), [`Norm`] (a LayerNorm),
//! [`FeedForward`] (two affine maps around an exact GELU) and
//! [`Attention`] (unmasked multi-head attention, torch
//! `nn.MultiheadAttention` semantics).

use super::config::LAYER_NORM_EPS;
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `x W^T + b`, the weight held transposed.
pub struct Affine<R: Runtime> {
    /// `[in, out]`, contiguous.
    pub(super) weight_t: Tensor<R>,
    /// `[out]`, or none.
    pub(super) bias: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Affine<R> {
    /// `x` `[N, in]` to `[N, out]`.
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        Ok(match &self.bias {
            Some(bias) => client.matmul_bias(x, &self.weight_t, bias)?,
            None => client.matmul(x, &self.weight_t)?,
        })
    }

    /// The output width.
    pub(super) fn out_features(&self) -> usize {
        self.weight_t.shape()[1]
    }
}

/// A LayerNorm over the last dim with weight and bias, eps
/// [`LAYER_NORM_EPS`].
pub struct Norm<R: Runtime> {
    pub(super) weight: Tensor<R>,
    pub(super) bias: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Norm<R> {
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        Ok(client.layer_norm(x, &self.weight, &self.bias, LAYER_NORM_EPS)?)
    }
}

/// `outer(GELU(inner(x)))`, GELU in its exact erf form.
pub struct FeedForward<R: Runtime> {
    pub(super) inner: Affine<R>,
    pub(super) outer: Affine<R>,
}

impl<R: Runtime<DType = DType>> FeedForward<R> {
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let hidden = client.gelu_erf(&self.inner.forward(client, x)?)?;
        self.outer.forward(client, &hidden)
    }
}

/// Unmasked multi-head attention: `heads` heads of `width / heads`, scores
/// scaled by `1 / sqrt(width / heads)`, heads concatenated, then `out`.
pub struct Attention<R: Runtime> {
    pub(super) q: Affine<R>,
    pub(super) k: Affine<R>,
    pub(super) v: Affine<R>,
    pub(super) out: Affine<R>,
    pub(super) heads: usize,
}

impl<R: Runtime<DType = DType>> Attention<R> {
    /// `query` `[N, W]` attends to `key_value` `[L, W]`, every query row to
    /// every key row; returns `[N, W]`.
    pub(super) fn forward<C: ModelClient<R>>(
        &self,
        client: &C,
        query: &Tensor<R>,
        key_value: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let n = query.shape()[0];
        let l = key_value.shape()[0];
        let width = self.q.out_features();
        let h = self.heads;
        let d = width / h;
        let q = self
            .q
            .forward(client, query)?
            .reshape(&[n, h, d])?
            .permute(&[1, 0, 2])?
            .contiguous()?;
        let k_t = self
            .k
            .forward(client, key_value)?
            .reshape(&[l, h, d])?
            .permute(&[1, 2, 0])?
            .contiguous()?;
        let v = self
            .v
            .forward(client, key_value)?
            .reshape(&[l, h, d])?
            .permute(&[1, 0, 2])?
            .contiguous()?;
        let scores = client.mul_scalar(&client.matmul(&q, &k_t)?, 1.0 / (d as f64).sqrt())?;
        let weights = client.softmax(&scores, -1)?;
        let heads = client
            .matmul(&weights, &v)?
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[n, width])?;
        self.out.forward(client, &heads)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use std::collections::HashMap;

    /// Host weights by state-dict name, f64.
    pub(crate) type HostWeights = HashMap<String, Vec<f64>>;

    /// Deterministic values in `[-0.5, 0.5)`.
    pub(crate) fn values(n: usize, seed: u64) -> Vec<f32> {
        (0..n as u64)
            .map(|i| {
                let x = (i + 1).wrapping_mul(seed.wrapping_mul(2_654_435_761) | 1) % 1009;
                x as f32 / 1009.0 - 0.5
            })
            .collect()
    }

    /// `erf(v)` by the midpoint rule over `exp(-t^2)`.
    pub(crate) fn erf(v: f64) -> f64 {
        let steps = 4_000;
        let h = v / steps as f64;
        let sum: f64 = (0..steps)
            .map(|i| {
                let t = (i as f64 + 0.5) * h;
                (-t * t).exp()
            })
            .sum();
        2.0 / std::f64::consts::PI.sqrt() * sum * h
    }

    /// GELU in its exact erf form.
    pub(crate) fn gelu(v: f64) -> f64 {
        0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2))
    }

    /// `x [rows, input] W^T + b`, `W` `[out, input]` row-major.
    pub(crate) fn host_affine(
        x: &[f64],
        rows: usize,
        w: &[f64],
        b: Option<&[f64]>,
        input: usize,
    ) -> Vec<f64> {
        let out = w.len() / input;
        (0..rows * out)
            .map(|n| {
                let (r, o) = (n / out, n % out);
                let dot: f64 = (0..input)
                    .map(|i| x[r * input + i] * w[o * input + i])
                    .sum();
                dot + b.map_or(0.0, |b| b[o])
            })
            .collect()
    }

    /// The affine map `{p}.weight` (and `{p}.bias` when `bias`).
    pub(crate) fn host_linear(
        m: &HostWeights,
        x: &[f64],
        rows: usize,
        p: &str,
        input: usize,
        bias: bool,
    ) -> Vec<f64> {
        let b = bias.then(|| m[&format!("{p}.bias")].as_slice());
        host_affine(x, rows, &m[&format!("{p}.weight")], b, input)
    }

    /// The LayerNorm `{p}` over rows of its width, eps 1e-5.
    pub(crate) fn host_norm(m: &HostWeights, x: &[f64], rows: usize, p: &str) -> Vec<f64> {
        let w = &m[&format!("{p}.weight")];
        let b = &m[&format!("{p}.bias")];
        let d = w.len();
        let mut out = Vec::with_capacity(rows * d);
        for row in x.chunks(d).take(rows) {
            let mean = row.iter().sum::<f64>() / d as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            out.extend((0..d).map(|i| (row[i] - mean) * inv * w[i] + b[i]));
        }
        out
    }

    /// `outer(GELU(inner(x)))`.
    pub(crate) fn host_ff(
        m: &HostWeights,
        x: &[f64],
        rows: usize,
        inner: &str,
        outer: &str,
        input: usize,
    ) -> Vec<f64> {
        let hidden: Vec<f64> = host_linear(m, x, rows, inner, input, true)
            .into_iter()
            .map(gelu)
            .collect();
        let mid = hidden.len() / rows;
        host_linear(m, &hidden, rows, outer, mid, true)
    }

    /// The torch `nn.MultiheadAttention` `{p}`: `xq` `[n, W]` attends to
    /// `xkv` `[l, W]` over `heads` heads.
    pub(crate) fn host_mha(
        m: &HostWeights,
        xq: &[f64],
        n: usize,
        xkv: &[f64],
        l: usize,
        p: &str,
        heads: usize,
    ) -> Vec<f64> {
        let iw = &m[&format!("{p}.in_proj_weight")];
        let ib = &m[&format!("{p}.in_proj_bias")];
        let w = ib.len() / 3;
        let d = w / heads;
        let part = |x: &[f64], rows: usize, i: usize| {
            let wi = &iw[i * w * w..(i + 1) * w * w];
            host_affine(x, rows, wi, Some(&ib[i * w..(i + 1) * w]), w)
        };
        let (q, k, v) = (part(xq, n, 0), part(xkv, l, 1), part(xkv, l, 2));
        let mut concat = vec![0.0; n * w];
        for hd in 0..heads {
            for r in 0..n {
                let s: Vec<f64> = (0..l)
                    .map(|c| {
                        let dot: f64 = (0..d)
                            .map(|e| q[r * w + hd * d + e] * k[c * w + hd * d + e])
                            .sum();
                        dot / (d as f64).sqrt()
                    })
                    .collect();
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
                for e in 0..d {
                    concat[r * w + hd * d + e] = (0..l)
                        .map(|c| (s[c] - top).exp() / z * v[c * w + hd * d + e])
                        .sum();
                }
            }
        }
        host_linear(m, &concat, n, &format!("{p}.out_proj"), w, true)
    }

    /// A device affine map from host `w` `[out, input]` and `b`.
    fn device_affine(device: &CpuDevice, w: &[f64], b: &[f64], input: usize) -> Affine<CpuRuntime> {
        let out = b.len();
        let w_t: Vec<f32> = (0..input * out)
            .map(|n| w[(n % out) * input + n / out] as f32)
            .collect();
        let b: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        Affine {
            weight_t: Tensor::<CpuRuntime>::from_slice(&w_t, &[input, out], device).unwrap(),
            bias: Some(Tensor::<CpuRuntime>::from_slice(&b, &[out], device).unwrap()),
        }
    }

    fn host_values(n: usize, seed: u64) -> Vec<f64> {
        values(n, seed).iter().map(|&v| f64::from(v)).collect()
    }

    fn close(got: &[f32], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((f64::from(*g) - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    /// Attention over `[3, 8]` queries and `[5, 8]` keys, two heads,
    /// against the host evaluation.
    #[test]
    fn attention_matches_host() {
        let (client, device) = cpu_setup();
        let (w, heads, n, l) = (8usize, 2usize, 3usize, 5usize);
        let mut m = HostWeights::new();
        m.insert("a.in_proj_weight".into(), host_values(3 * w * w, 11));
        m.insert("a.in_proj_bias".into(), host_values(3 * w, 13));
        m.insert("a.out_proj.weight".into(), host_values(w * w, 17));
        m.insert("a.out_proj.bias".into(), host_values(w, 19));
        let (iw, ib) = (&m["a.in_proj_weight"], &m["a.in_proj_bias"]);
        let part = |i: usize| {
            device_affine(
                &device,
                &iw[i * w * w..(i + 1) * w * w],
                &ib[i * w..(i + 1) * w],
                w,
            )
        };
        let attention = Attention {
            q: part(0),
            k: part(1),
            v: part(2),
            out: device_affine(&device, &m["a.out_proj.weight"], &m["a.out_proj.bias"], w),
            heads,
        };
        let xq = values(n * w, 23);
        let xk = values(l * w, 29);
        let query = Tensor::<CpuRuntime>::from_slice(&xq, &[n, w], &device).unwrap();
        let memory = Tensor::<CpuRuntime>::from_slice(&xk, &[l, w], &device).unwrap();
        let got = attention.forward(&client, &query, &memory).unwrap();
        let to64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        let want = host_mha(&m, &to64(&xq), n, &to64(&xk), l, "a", heads);
        close(&got.to_vec::<f32>(), &want);
    }

    /// The exact-erf GELU between two affine maps.
    #[test]
    fn feed_forward_uses_exact_gelu() {
        let (client, device) = cpu_setup();
        let mut m = HostWeights::new();
        m.insert("i.weight".into(), host_values(6 * 4, 31));
        m.insert("i.bias".into(), host_values(6, 32));
        m.insert("o.weight".into(), host_values(4 * 6, 37));
        m.insert("o.bias".into(), host_values(4, 38));
        let ff = FeedForward {
            inner: device_affine(&device, &m["i.weight"], &m["i.bias"], 4),
            outer: device_affine(&device, &m["o.weight"], &m["o.bias"], 6),
        };
        let x = values(2 * 4, 41);
        let xt = Tensor::<CpuRuntime>::from_slice(&x, &[2, 4], &device).unwrap();
        let got = ff.forward(&client, &xt).unwrap().to_vec::<f32>();
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        close(&got, &host_ff(&m, &x, 2, "i", "o", 4));
    }
}

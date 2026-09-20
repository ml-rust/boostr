//! `qwen35` gated attention inference forward over a [`KvCache`].
//!
//! Ports `build_layer_attn` (`src/models/qwen35.cpp`) of llama.cpp
//! step for step:
//!
//! 1. `q_full = attn_q(x)`, split per head into query and gate
//! 2. `q = rms_norm(q, attn_q_norm)`, `k = rms_norm(attn_k(x), attn_k_norm)`, `v = attn_v(x)`
//! 3. IMROPE on q and k (`ggml_rope_multi`, `GGML_ROPE_TYPE_IMROPE`)
//! 4. KV cache append, causal GQA attention at `1 / sqrt(head_dim)`
//! 5. `attn ⊙ sigmoid(gate)`, then `attn_output`

use super::layer::Qwen35AttentionBlock;
use super::projections::Projections;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::attention_mask::causal_window_mask;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_permute, var_reshape, var_sigmoid_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, FwhtOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Inference forward. `x` is the `attn_norm`-ed hidden state
    /// `[batch, seq, hidden_size]`; the result is the `attn_output`
    /// projection, `[batch, seq, hidden_size]`, without the residual.
    ///
    /// - `rope`: `[max_pos, rope_dim / 2]` table, see
    ///   `Qwen35AttentionConfig::rope_table`
    /// - `positions`: `[4, seq]` integer IMROPE streams `t, h, w, e`
    /// - `kv_cache`: holds the left context; the new k/v rows are appended.
    ///   The causal mask offsets queries by `kv_cache.seq_len()` before the
    ///   append, so `seq > 1` is a prefill and `seq == 1` a decode step.
    ///
    /// The output is a detached `Var`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]` or
    /// the rope table width is not `rope_dim / 2`; cache and op errors
    /// propagate.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
        kv_cache: &mut KvCache<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + FwhtOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let cfg = &self.cfg;
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[2] != cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention: expected [batch, seq, {}], got {shape:?}",
                    cfg.hidden_size
                ),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);
        let (h, hd) = (cfg.num_heads, cfg.head_dim);
        let half_rot = cfg.rope_dim / 2;
        if rope.cos_cache().shape().get(1) != Some(&half_rot) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention: rope table must be [max_pos, {half_rot}], got {:?}",
                    rope.cos_cache().shape()
                ),
            });
        }

        // 1-2. `attn_q`/`attn_k`/`attn_v` projections (fused via
        // `forward_batch` — any shared Hadamard rotation runs once), joint
        // query/gate split, per-head q/k RMS norm. See `super::projections`.
        let Projections { q, gate, k, v } = self.project(client, x, batch, seq)?;

        // 3. IMROPE on [batch, seq, heads, head_dim]. `mrope_selector` is
        // built once in `new()` and reused for both q and k.
        let (cos, sin) = (rope.cos_cache(), rope.sin_cache());
        let n_rot = cfg.rope_dim;
        let q =
            client.mrope_interleaved_fused(&q, cos, sin, positions, &self.mrope_selector, n_rot)?;
        let k =
            client.mrope_interleaved_fused(&k, cos, sin, positions, &self.mrope_selector, n_rot)?;

        // 4. [B, S, H, D] -> [B, H, S, D]; append; causal GQA attention.
        let q = var_contiguous(&var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let k = var_contiguous(&var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let v = var_contiguous(&var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;

        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;
        let (cached_k, cached_v) = kv_cache.get_kv()?;
        let cached_k = Var::new(cached_k.contiguous()?, false);
        let cached_v = Var::new(cached_v.contiguous()?, false);
        let rep = cfg.head_repeat();
        let cached_k = repeat_kv(&cached_k, rep).map_err(Error::Numr)?;
        let cached_v = repeat_kv(&cached_v, rep).map_err(Error::Numr)?;

        let sk = kv_cache.seq_len();
        let (dtype, device) = (q.tensor().dtype(), q.tensor().device());
        let mask = Var::new(
            causal_window_mask(client, seq, sk, 0, dtype, device)?,
            false,
        );
        let attn = multi_head_attention_impl(client, &q, &cached_k, &cached_v, Some(&mask), h)?;

        // 5. [B, H, S, D] -> [B, S, H, D]; gate; flatten heads; project.
        let attn = var_contiguous(&var_permute(&attn, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let gated = var_sigmoid_mul(&gate, &attn, client).map_err(Error::Numr)?;
        let gated = var_reshape(&gated, &[batch, seq, h * hd]).map_err(Error::Numr)?;
        let out = self.attn_output.forward(client, &gated)?;
        Ok(Var::new(out.tensor().clone(), false))
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::Qwen35AttentionWeights;
    use super::*;
    use crate::model::config::Qwen35AttentionConfig;
    use crate::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const HIDDEN: usize = 8;
    const H: usize = 2;
    const H_KV: usize = 1;
    const HD: usize = 8;
    const ROPE_DIM: usize = 4;
    const SEQ: usize = 5;
    const MAX_POS: usize = 16;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }

        fn data(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.uniform(-scale, scale)).collect()
        }

        fn tensor(
            &mut self,
            device: &CpuDevice,
            shape: &[usize],
            scale: f32,
        ) -> Tensor<CpuRuntime> {
            let n: usize = shape.iter().product();
            Tensor::<CpuRuntime>::from_slice(&self.data(n, scale), shape, device).unwrap()
        }
    }

    fn cfg(sections: [usize; 4]) -> Qwen35AttentionConfig {
        Qwen35AttentionConfig {
            hidden_size: HIDDEN,
            num_heads: H,
            num_kv_heads: H_KV,
            head_dim: HD,
            rope_dim: ROPE_DIM,
            rope_sections: sections,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }
    }

    fn linear(w: Tensor<CpuRuntime>) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(Linear::new(w, None, false)))
    }

    /// Deterministic weights. `attn_q` rows and `attn_output` come back as
    /// host vectors so probes can edit them before building.
    struct Parts {
        q: Vec<f32>,
        k: Tensor<CpuRuntime>,
        v: Tensor<CpuRuntime>,
        o: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Tensor<CpuRuntime>,
    }

    fn parts(device: &CpuDevice, seed: u64) -> Parts {
        let mut rng = Lcg(seed);
        let s = 0.5 / (HIDDEN as f32).sqrt();
        Parts {
            q: rng.data(H * 2 * HD * HIDDEN, s),
            k: rng.tensor(device, &[H_KV * HD, HIDDEN], s),
            v: rng.tensor(device, &[H_KV * HD, HIDDEN], s),
            o: rng.data(HIDDEN * H * HD, 0.5 / ((H * HD) as f32).sqrt()),
            q_norm: rng.data(HD, 1.0).iter().map(|x| x + 1.5).collect(),
            k_norm: rng.tensor(device, &[HD], 1.0),
        }
    }

    fn build(
        device: &CpuDevice,
        sections: [usize; 4],
        p: Parts,
    ) -> Qwen35AttentionBlock<CpuRuntime> {
        let t = |d: &[f32], shape: &[usize]| {
            Tensor::<CpuRuntime>::from_slice(d, shape, device).unwrap()
        };
        let weights = Qwen35AttentionWeights {
            attn_q: linear(t(&p.q, &[H * 2 * HD, HIDDEN])),
            attn_k: linear(p.k),
            attn_v: linear(p.v),
            attn_output: linear(t(&p.o, &[HIDDEN, H * HD])),
            attn_q_norm: t(&p.q_norm, &[HD]),
            attn_k_norm: p.k_norm,
        };
        Qwen35AttentionBlock::new(cfg(sections), weights).unwrap()
    }

    fn cache(device: &CpuDevice) -> KvCache<CpuRuntime> {
        KvCache::<CpuRuntime>::new(1, H_KV, MAX_POS, MAX_POS, HD, DType::F32, device).unwrap()
    }

    fn rope(device: &CpuDevice) -> RoPE<CpuRuntime> {
        cfg([1, 1, 0, 0])
            .rope_table::<CpuRuntime>(MAX_POS, device)
            .unwrap()
    }

    /// `[4, seq]` i32 text positions: every token at `kv_cache.seq_len() + i`
    /// on `t`, `h`, `w`; `0` on `e`.
    fn text_positions(
        seq: usize,
        cache: &KvCache<CpuRuntime>,
        device: &CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let start = cache.seq_len();
        let mut data = Vec::with_capacity(4 * seq);
        for _ in 0..3 {
            data.extend((0..seq).map(|i| (start + i) as i32));
        }
        data.resize(4 * seq, 0);
        Tensor::<CpuRuntime>::from_slice(&data, &[4, seq], device).unwrap()
    }

    fn run(
        client: &CpuClient,
        block: &Qwen35AttentionBlock<CpuRuntime>,
        rope: &RoPE<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        cache: &mut KvCache<CpuRuntime>,
    ) -> Tensor<CpuRuntime> {
        let positions = text_positions(x.shape()[1], cache, x.device());
        block
            .forward(client, &Var::new(x.clone(), false), rope, &positions, cache)
            .unwrap()
            .tensor()
            .clone()
    }

    fn max_abs_diff(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
        let a = a.to_vec::<f32>();
        let b = b.to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Prefill of 5 tokens equals prefill of 3 then 2 cached decode steps.
    fn check_cache_carry(sections: [usize; 4]) {
        let (client, device) = cpu_setup();
        let block = build(&device, sections, parts(&device, 0x3500_0001));
        let rope = rope(&device);
        let x = Lcg(11).tensor(&device, &[1, SEQ, HIDDEN], 1.0);

        let mut full_cache = cache(&device);
        let full = run(&client, &block, &rope, &x, &mut full_cache);
        assert_eq!(full.shape(), &[1, SEQ, HIDDEN]);

        let mut step_cache = cache(&device);
        let head = x.narrow(1, 0, 3).unwrap().contiguous().unwrap();
        let mut chunks = vec![run(&client, &block, &rope, &head, &mut step_cache)];
        for t in 3..SEQ {
            let xt = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            chunks.push(run(&client, &block, &rope, &xt, &mut step_cache));
        }
        let refs: Vec<&Tensor<CpuRuntime>> = chunks.iter().collect();
        let stepped = client.cat(&refs, 1).unwrap();

        let diff = max_abs_diff(&full, &stepped);
        assert!(diff < 1e-4, "sections {sections:?}: output diff {diff}");
        assert_eq!(full_cache.seq_len(), step_cache.seq_len());
        let (fk, _) = full_cache.get_kv().unwrap();
        let (sk, _) = step_cache.get_kv().unwrap();
        let k_diff = max_abs_diff(&fk.contiguous().unwrap(), &sk.contiguous().unwrap());
        assert!(k_diff < 1e-5, "k cache diff {k_diff}");
    }

    #[test]
    fn prefill_matches_prefill_then_decode() {
        check_cache_carry([1, 1, 0, 0]);
    }

    #[test]
    fn prefill_matches_prefill_then_decode_with_e_sector() {
        check_cache_carry([1, 0, 1, 0]);
    }

    /// Set head 0's gate rows of `attn_q` to `sign * 1e3` and feed a positive
    /// input, so `sigmoid(gate)` for head 0 is exactly `0` (`sign < 0`) or
    /// `1` (`sign > 0`). Return outputs with two different head-0 column
    /// blocks of `attn_output`.
    fn gated_pair(sign: f32) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let (client, device) = cpu_setup();
        let rope = rope(&device);
        let x = Tensor::<CpuRuntime>::from_slice(
            &(0..SEQ * HIDDEN)
                .map(|i| 0.5 + (i % 5) as f32 * 0.1)
                .collect::<Vec<_>>(),
            &[1, SEQ, HIDDEN],
            &device,
        )
        .unwrap();

        let mut outs = Vec::new();
        for variant in 0..2 {
            let mut p = parts(&device, 0x3500_0002);
            // Head 0 gate rows: output rows [HD, 2*HD).
            for row in HD..2 * HD {
                for col in 0..HIDDEN {
                    p.q[row * HIDDEN + col] = sign * 1e3;
                }
            }
            if variant == 1 {
                // Head 0 columns of attn_output: [0, HD) in every row.
                for row in 0..HIDDEN {
                    for col in 0..HD {
                        p.o[row * H * HD + col] += 0.7;
                    }
                }
            }
            let block = build(&device, [1, 1, 0, 0], p);
            let mut c = cache(&device);
            outs.push(run(&client, &block, &rope, &x, &mut c));
        }
        let b = outs.pop().unwrap();
        let a = outs.pop().unwrap();
        (a, b)
    }

    #[test]
    fn zero_gate_removes_head_from_output_projection() {
        let (a, b) = gated_pair(-1.0);
        assert!(
            max_abs_diff(&a, &b) < 1e-6,
            "gated-off head still reaches attn_output"
        );
    }

    #[test]
    fn open_gate_keeps_head_in_output_projection() {
        let (a, b) = gated_pair(1.0);
        assert!(
            max_abs_diff(&a, &b) > 1e-3,
            "control: open gate must expose head 0"
        );
    }

    #[test]
    fn q_norm_weight_changes_output() {
        let (client, device) = cpu_setup();
        let rope = rope(&device);
        let x = Lcg(5).tensor(&device, &[1, SEQ, HIDDEN], 1.0);

        let base = build(&device, [1, 1, 0, 0], parts(&device, 0x3500_0003));
        let mut p = parts(&device, 0x3500_0003);
        for w in &mut p.q_norm {
            *w *= 3.0;
        }
        let probe = build(&device, [1, 1, 0, 0], p);

        let mut c1 = cache(&device);
        let mut c2 = cache(&device);
        let a = run(&client, &base, &rope, &x, &mut c1);
        let b = run(&client, &probe, &rope, &x, &mut c2);
        assert!(max_abs_diff(&a, &b) > 1e-4);
    }

    #[test]
    fn rejects_wrong_rope_table_width() {
        let (client, device) = cpu_setup();
        let block = build(&device, [1, 1, 0, 0], parts(&device, 0x3500_0004));
        let wide =
            RoPE::<CpuRuntime>::precompute_freqs(MAX_POS, HD, 10_000.0, None, &device).unwrap();
        let x = Lcg(1).tensor(&device, &[1, 2, HIDDEN], 1.0);
        let mut c = cache(&device);
        let positions = text_positions(2, &c, &device);
        assert!(
            block
                .forward(&client, &Var::new(x, false), &wide, &positions, &mut c)
                .is_err()
        );
    }

    #[test]
    fn rejects_bad_input_shape() {
        let (client, device) = cpu_setup();
        let block = build(&device, [1, 1, 0, 0], parts(&device, 0x3500_0005));
        let rope = rope(&device);
        let x = Lcg(1).tensor(&device, &[1, 2, HIDDEN + 1], 1.0);
        let mut c = cache(&device);
        let positions = text_positions(2, &c, &device);
        assert!(
            block
                .forward(&client, &Var::new(x, false), &rope, &positions, &mut c)
                .is_err()
        );
    }
}

//! Step 1-2 of [`Qwen35AttentionBlock::forward`]: `attn_q`/`attn_k`/`attn_v`
//! projections (fused via [`MaybeRotatedLinear::forward_batch`]), the joint
//! query/gate split, and per-head q/k RMS norm. Covered by its own inline
//! tests for the rotation-batching behavior and the query/gate split.
//!
//! `attn_q` is stored regrouped (see `layer`), so its output is
//! `[query: num_heads * head_dim | gate: num_heads * head_dim]` and each
//! half is a `narrow` at a column boundary. For one token that view is
//! already dense — the query at offset 0, the gate at offset
//! `num_heads * head_dim` — and goes on without a copy; a multi-token
//! projection is strided along `seq` and each half is copied dense once.

use super::layer::Qwen35AttentionBlock;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::MaybeRotatedLinear;
use crate::nn::var_ops::var_contiguous;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, FwhtOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// `q`, `gate`, `k`, `v`, each `[batch, seq, heads, head_dim]` (`q`/`gate`
/// with `heads = num_heads`, `k`/`v` with `heads = num_kv_heads`), `q`/`k`
/// already through `attn_q_norm`/`attn_k_norm`.
pub(super) struct Projections<R: Runtime> {
    pub(super) q: Var<R>,
    pub(super) gate: Var<R>,
    pub(super) k: Var<R>,
    pub(super) v: Var<R>,
}

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Runs `attn_q`/`attn_k`/`attn_v` (sharing `x`) through one
    /// `forward_batch` call, splits `attn_q`'s output into query and gate
    /// per head, reshapes `k`/`v` per head, and applies `attn_q_norm`/
    /// `attn_k_norm`.
    pub(super) fn project<C>(
        &self,
        client: &C,
        x: &Var<R>,
        batch: usize,
        seq: usize,
    ) -> Result<Projections<R>>
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
        let (h, h_kv, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);

        let mut projected = MaybeRotatedLinear::forward_batch(
            &[&self.attn_q, &self.attn_k, &self.attn_v],
            client,
            x,
        )?
        .into_iter();
        let q_full = projected.next().ok_or_else(|| Error::ModelError {
            reason: "qwen35_attention: forward_batch returned no attn_q output".to_string(),
        })?;
        let k = projected.next().ok_or_else(|| Error::ModelError {
            reason: "qwen35_attention: forward_batch returned no attn_k output".to_string(),
        })?;
        let v = projected.next().ok_or_else(|| Error::ModelError {
            reason: "qwen35_attention: forward_batch returned no attn_v output".to_string(),
        })?;

        let q_dim = h * hd;
        let q_full = var_reshape(&q_full, &[batch, seq, 2 * q_dim]).map_err(Error::Numr)?;
        let q = split_heads(&q_full, 0, batch, seq, h, hd)?;
        let gate = split_heads(&q_full, q_dim, batch, seq, h, hd)?;

        let k = var_reshape(&k, &[batch, seq, h_kv, hd]).map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq, h_kv, hd]).map_err(Error::Numr)?;
        let (q, k) = self.qk_norm.forward(client, &q, &k)?;

        Ok(Projections { q, gate, k, v })
    }
}

/// Columns `[start, start + heads * head_dim)` of `x` (`[batch, seq, 2 *
/// heads * head_dim]`) as `[batch, seq, heads, head_dim]`. A view that is
/// already dense in its stride pattern is reshaped in place, offset and
/// all; any other is copied dense first.
fn split_heads<R: Runtime<DType = DType>>(
    x: &Var<R>,
    start: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Var<R>>
where
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let half = var_narrow(x, -1, start, heads * head_dim).map_err(Error::Numr)?;
    let half = var_contiguous(&half)?;
    var_reshape(&half, &[batch, seq, heads, head_dim]).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::layer::Qwen35AttentionWeights;
    use super::*;
    use crate::model::config::Qwen35AttentionConfig;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::RotatedLinear;
    use crate::nn::linear::dense::Linear;
    use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HIDDEN: usize = 8;
    const H: usize = 2;
    const H_KV: usize = 1;
    const HD: usize = 8;

    fn tensor(device: &CpuDevice, shape: &[usize], scale: f32, seed: u32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
                scale * ((h % 1000) as f32 / 1000.0 - 0.5)
            })
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn plain(w: Tensor<CpuRuntime>) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(Linear::new(w, None, false)))
    }

    fn rotated(
        w: Tensor<CpuRuntime>,
        rot: HadamardRotation<CpuRuntime>,
    ) -> MaybeRotatedLinear<CpuRuntime> {
        RotatedLinear::new(MaybeQuantLinear::Standard(Linear::new(w, None, false)), rot)
            .unwrap()
            .into()
    }

    fn rotation(device: &CpuDevice) -> HadamardRotation<CpuRuntime> {
        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, device).unwrap()
    }

    /// `attn_q` as stored: `[head][query | gate]`, rotated.
    fn stored_attn_q(device: &CpuDevice) -> MaybeRotatedLinear<CpuRuntime> {
        rotated(
            tensor(device, &[H * 2 * HD, HIDDEN], 0.1, 1),
            rotation(device),
        )
    }

    fn block_with_rotated_projections(device: &CpuDevice) -> Qwen35AttentionBlock<CpuRuntime> {
        let rot = rotation(device);
        let weights = Qwen35AttentionWeights {
            attn_q: rotated(tensor(device, &[H * 2 * HD, HIDDEN], 0.1, 1), rot.clone()),
            attn_k: rotated(tensor(device, &[H_KV * HD, HIDDEN], 0.1, 2), rot.clone()),
            attn_v: rotated(tensor(device, &[H_KV * HD, HIDDEN], 0.1, 3), rot),
            attn_output: plain(tensor(device, &[HIDDEN, H * HD], 0.1, 4)),
            attn_q_norm: tensor(device, &[HD], 1.0, 5),
            attn_k_norm: tensor(device, &[HD], 1.0, 6),
        };
        let cfg = Qwen35AttentionConfig {
            hidden_size: HIDDEN,
            num_heads: H,
            num_kv_heads: H_KV,
            head_dim: HD,
            rope_dim: 4,
            rope_sections: [1, 1, 0, 0],
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        };
        Qwen35AttentionBlock::new(cfg, weights).unwrap()
    }

    /// `attn_q`/`attn_k`/`attn_v` (`Rotated`, sharing one rotation) via
    /// `project` (`forward_batch` internally) must match three individual
    /// `forward` calls — the pre-batching behavior — bit-for-bit. The
    /// query and gate references come from the STORED interleaved `attn_q`
    /// split per head, so the constructor's row regrouping is checked too.
    #[test]
    fn project_over_shared_rotation_matches_per_projection_forward() {
        let (client, device) = cpu_setup();
        let block = block_with_rotated_projections(&device);
        let x = Var::new(tensor(&device, &[1, 5, HIDDEN], 1.0, 9), false);

        let via_project = block.project(&client, &x, 1, 5).unwrap();

        let q_solo = stored_attn_q(&device).forward(&client, &x).unwrap();
        let k_solo = block.attn_k.forward(&client, &x).unwrap();
        let v_solo = block.attn_v.forward(&client, &x).unwrap();
        let q_solo = var_reshape(&q_solo, &[1, 5, H, 2 * HD]).unwrap();
        let gate_solo = var_contiguous(&var_narrow(&q_solo, -1, HD, HD).unwrap()).unwrap();
        let q_solo = var_contiguous(&var_narrow(&q_solo, -1, 0, HD).unwrap()).unwrap();
        let k_solo = var_reshape(&k_solo, &[1, 5, H_KV, HD]).unwrap();
        let v_solo = var_reshape(&v_solo, &[1, 5, H_KV, HD]).unwrap();
        let (q_solo, k_solo) = block.qk_norm.forward(&client, &q_solo, &k_solo).unwrap();

        assert_eq!(
            via_project.q.tensor().to_vec::<f32>(),
            q_solo.tensor().to_vec::<f32>()
        );
        assert_eq!(via_project.gate.shape(), &[1, 5, H, HD]);
        assert_eq!(
            via_project
                .gate
                .tensor()
                .contiguous()
                .unwrap()
                .to_vec::<f32>(),
            gate_solo.tensor().to_vec::<f32>()
        );
        assert_eq!(
            via_project.k.tensor().to_vec::<f32>(),
            k_solo.tensor().to_vec::<f32>()
        );
        assert_eq!(
            via_project.v.tensor().to_vec::<f32>(),
            v_solo.tensor().to_vec::<f32>()
        );
    }

    /// One token: the gate is the projection's own storage at column
    /// offset `H * HD`, not a copy.
    #[test]
    fn single_token_gate_is_an_offset_view() {
        let (client, device) = cpu_setup();
        let block = block_with_rotated_projections(&device);
        let x = Var::new(tensor(&device, &[1, 1, HIDDEN], 1.0, 9), false);
        let Projections { gate, .. } = block.project(&client, &x, 1, 1).unwrap();
        assert_eq!(gate.shape(), &[1, 1, H, HD]);
        assert_eq!(gate.tensor().offset(), H * HD);
        assert!(gate.tensor().is_contiguous());
    }

    #[test]
    fn project_rejects_mismatched_rotations() {
        let (client, device) = cpu_setup();
        let signs_a: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let signs_b: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();
        let rot_a =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs_a), DType::F32, &device)
                .unwrap();
        let rot_b =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs_b), DType::F32, &device)
                .unwrap();

        let weights = Qwen35AttentionWeights {
            attn_q: rotated(tensor(&device, &[H * 2 * HD, HIDDEN], 0.1, 1), rot_a),
            attn_k: rotated(tensor(&device, &[H_KV * HD, HIDDEN], 0.1, 2), rot_b),
            attn_v: plain(tensor(&device, &[H_KV * HD, HIDDEN], 0.1, 3)),
            attn_output: plain(tensor(&device, &[HIDDEN, H * HD], 0.1, 4)),
            attn_q_norm: tensor(&device, &[HD], 1.0, 5),
            attn_k_norm: tensor(&device, &[HD], 1.0, 6),
        };
        let cfg = Qwen35AttentionConfig {
            hidden_size: HIDDEN,
            num_heads: H,
            num_kv_heads: H_KV,
            head_dim: HD,
            rope_dim: 4,
            rope_sections: [1, 1, 0, 0],
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        };
        let block = Qwen35AttentionBlock::new(cfg, weights).unwrap();
        let x = Var::new(tensor(&device, &[1, 5, HIDDEN], 1.0, 9), false);
        assert!(block.project(&client, &x, 1, 5).is_err());
    }
}

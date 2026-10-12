//! Unmerged LoRA adapters on a [`Qwen35AttentionBlock`]'s projections.
//!
//! Factors arrive in the stored layout of [`Qwen35AttentionWeights`](super::Qwen35AttentionWeights):
//! an `attn_q` adapter's `lora_b` rows interleave query and gate per head,
//! as the stored `attn_q` weight does. The block holds `attn_q` with its
//! rows regrouped to `[query | gate]` (see `layer`), so the `attn_q`
//! adapter's `lora_b` rows are regrouped the same way before the attach.
//! `lora_a` reads the input and stays as given.

use super::layer::Qwen35AttentionBlock;
use crate::error::Result;
use crate::nn::LoraFactors;
use crate::nn::linear::regroup_leading_axis;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

/// One [`Qwen35AttentionBlock`] projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35AttentionProjection {
    /// `attn_q`, `[num_heads * 2 * head_dim, hidden_size]`, query and gate
    /// interleaved per head in the adapter's layout.
    Q,
    /// `attn_k`, `[num_kv_heads * head_dim, hidden_size]`.
    K,
    /// `attn_v`, `[num_kv_heads * head_dim, hidden_size]`.
    V,
    /// `attn_output`, `[hidden_size, num_heads * head_dim]`.
    Output,
}

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Attach an unmerged adapter to `projection` through
    /// [`MaybeRotatedLinear::attach_factors`](crate::nn::MaybeRotatedLinear::attach_factors).
    /// `factors` follow the stored layout (see the module doc).
    /// `activation_dtype` is the dtype the block's input arrives in.
    ///
    /// # Errors
    ///
    /// As `attach_factors`: a Hadamard-rotated or already adapted
    /// projection, or a factor shape that does not fit it. An `attn_q`
    /// `lora_b` whose row count is not a multiple of `2 * num_heads` fails
    /// the regroup.
    pub fn attach_adapter<C: TypeConversionOps<R>>(
        &mut self,
        client: &C,
        projection: Qwen35AttentionProjection,
        factors: LoraFactors<R>,
        activation_dtype: DType,
    ) -> Result<()> {
        match projection {
            Qwen35AttentionProjection::Q => {
                let lora_b = regroup_leading_axis(&factors.lora_b, self.cfg.num_heads, 2)?;
                let regrouped = LoraFactors { lora_b, ..factors };
                self.attn_q
                    .attach_factors(client, regrouped, activation_dtype)
            }
            Qwen35AttentionProjection::K => {
                self.attn_k
                    .attach_factors(client, factors, activation_dtype)
            }
            Qwen35AttentionProjection::V => {
                self.attn_v
                    .attach_factors(client, factors, activation_dtype)
            }
            Qwen35AttentionProjection::Output => {
                self.attn_output
                    .attach_factors(client, factors, activation_dtype)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::Qwen35AttentionWeights;
    use super::*;
    use crate::model::config::Qwen35AttentionConfig;
    use crate::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::ops::{BinaryOps, MatmulOps, ScalarOps};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const HD: usize = 4;
    const RANK: usize = 3;
    const SCALING: f32 = 1.25;

    fn cfg() -> Qwen35AttentionConfig {
        Qwen35AttentionConfig {
            hidden_size: HIDDEN,
            num_heads: HEADS,
            num_kv_heads: 1,
            head_dim: HD,
            rope_dim: 4,
            rope_sections: [1, 1, 0, 0],
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }
    }

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.3)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn plain(w: &Tensor<CpuRuntime>) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(Linear::new(
            w.clone(),
            None,
            false,
        )))
    }

    /// An `attn_q` adapter in the stored, per-head interleaved layout
    /// acts as the merged stored weight does: `attn_q`'s output equals
    /// `x (W + s B A)^T` with the merged rows regrouped as `new` regroups
    /// the base.
    #[test]
    fn q_adapter_rows_follow_the_base_regroup() {
        let (client, device) = cpu_setup();
        let cfg = cfg();
        let q_rows = cfg.q_gate_dim();
        let w_q = tensor(&device, &[q_rows, HIDDEN], 0.13);
        let weights = Qwen35AttentionWeights {
            attn_q: plain(&w_q),
            attn_k: plain(&tensor(&device, &[cfg.kv_dim(), HIDDEN], 0.17)),
            attn_v: plain(&tensor(&device, &[cfg.kv_dim(), HIDDEN], 0.19)),
            attn_output: plain(&tensor(&device, &[HIDDEN, cfg.q_dim()], 0.23)),
            attn_q_norm: Tensor::<CpuRuntime>::ones(&[HD], DType::F32, &device).unwrap(),
            attn_k_norm: Tensor::<CpuRuntime>::ones(&[HD], DType::F32, &device).unwrap(),
        };
        let mut block = Qwen35AttentionBlock::new(cfg.clone(), weights).unwrap();
        let a = tensor(&device, &[RANK, HIDDEN], 0.31);
        let b = tensor(&device, &[q_rows, RANK], 0.47);
        block
            .attach_adapter(
                &client,
                Qwen35AttentionProjection::Q,
                LoraFactors {
                    lora_a: a.clone(),
                    lora_b: b.clone(),
                    scaling: SCALING,
                },
                DType::F32,
            )
            .unwrap();
        assert!(block.attn_q.is_adapted());

        let ba = client.matmul(&b, &a).unwrap();
        let merged = client
            .add(&w_q, &client.mul_scalar(&ba, f64::from(SCALING)).unwrap())
            .unwrap();
        let merged = regroup_leading_axis(&merged, HEADS, 2).unwrap();
        let merged_t = merged.t().unwrap().contiguous().unwrap();
        let x = tensor(&device, &[3, HIDDEN], 0.61);
        let want: Vec<f32> = client.matmul(&x, &merged_t).unwrap().to_vec();
        let got: Vec<f32> = block
            .attn_q
            .forward(&client, &Var::new(x, false))
            .unwrap()
            .tensor()
            .to_vec();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }

        let second = block.attach_adapter(
            &client,
            Qwen35AttentionProjection::Q,
            LoraFactors {
                lora_a: a,
                lora_b: b,
                scaling: SCALING,
            },
            DType::F32,
        );
        assert!(second.is_err(), "a second adapter is refused");
    }
}

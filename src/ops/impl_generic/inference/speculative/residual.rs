//! The distribution `speculative_verify` draws `next` from, on the device.

use super::accept::LeadingAccepts;
use super::draw::{scalar_like, select};
use super::validate::VerifyShape;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, TensorOps, TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Row `pos[b]` of `t` for each `b`: `[B, N, V]` and `[B, 1]` F32 give `[B, 1, V]`.
fn row_at<R: Runtime<DType = DType>>(
    client: &R::Client,
    t: &Tensor<R>,
    pos: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R>,
{
    let (batch, vocab) = (t.shape()[0], t.shape()[2]);
    let index = client
        .cast(pos, DType::I32)?
        .reshape(&[batch, 1, 1])?
        .broadcast_to(&[batch, 1, vocab])?
        .contiguous()?;
    Ok(client.gather(t, 1, &index)?)
}

/// The `[B, 1, V]` distribution of `next`. K must be above 0.
///
/// - `m < K`: the residual `max(0, p_m - q_m)`. A one-hot `q_m` zeroes `p_m` at `d_m`.
/// - A zero residual falls back to `p_m`.
/// - `m == K`: `p_K`.
///
/// The row is not normalized. The draw scales its threshold by the row sum.
/// `target` must be contiguous. `ids` is `[1, 1, V]` F32 holding `0 .. V`.
pub fn next_distribution<R: Runtime<DType = DType>>(
    client: &R::Client,
    target: &Tensor<R>,
    draft_probs: Option<&Tensor<R>>,
    lead: &LeadingAccepts<R>,
    ids: &Tensor<R>,
    shape: VerifyShape,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let (batch, drafts) = (shape.batch, shape.drafts);
    let m = &lead.accepted;
    let p_row = row_at::<R>(client, target, m)?;
    let m_draft = client.minimum(m, &scalar_like(m, (drafts - 1) as f64)?)?;
    let residual = match draft_probs {
        Some(q) => {
            let q_row = row_at::<R>(client, &q.contiguous()?, &m_draft)?;
            let diff = client.sub(&p_row, &q_row)?;
            client.maximum(&diff, &scalar_like(&diff, 0.0)?)?
        }
        None => {
            let index = client.cast(&m_draft, DType::I32)?;
            let d_m = client
                .gather(&lead.ids, 1, &index)?
                .reshape(&[batch, 1, 1])?;
            let is_draft = client.eq(ids, &d_m)?;
            let keep = client.add_scalar(&client.mul_scalar(&is_draft, -1.0)?, 1.0)?;
            client.mul(&p_row, &keep)?
        }
    };
    let rejected = client
        .lt(m, &scalar_like(m, drafts as f64)?)?
        .reshape(&[batch, 1, 1])?;
    let mass = client.sum(&residual, &[2], true)?;
    let has_mass = client.gt(&mass, &scalar_like(&mass, 0.0)?)?;
    let use_residual = client.mul(&rejected, &has_mass)?;
    select::<R>(client, &use_residual, &residual, &p_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::UtilityOps;
    use numr::runtime::cpu::CpuRuntime;

    type T = Tensor<CpuRuntime>;

    /// One row, K = 1, V = 4, with `m` and `d_0` given.
    fn dist(p: [[f32; 4]; 2], q: Option<[f32; 4]>, d0: i32, m: f32) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let flat: Vec<f32> = p.iter().flatten().copied().collect();
        let target = T::from_slice(&flat, &[1, 2, 4], &device).expect("p");
        let q = q.map(|q| T::from_slice(&q, &[1, 1, 4], &device).expect("q"));
        let lead = LeadingAccepts {
            ids: T::from_slice(&[d0 as f32], &[1, 1], &device).expect("ids"),
            accepted: T::from_slice(&[m], &[1, 1], &device).expect("m"),
        };
        let ids = client
            .arange(0.0, 4.0, 1.0, DType::F32)
            .expect("ids")
            .reshape(&[1, 1, 4])
            .expect("ids shape");
        let shape = VerifyShape {
            batch: 1,
            drafts: 1,
            vocab: 4,
        };
        next_distribution::<CpuRuntime>(&client, &target, q.as_ref(), &lead, &ids, shape)
            .expect("dist")
            .to_vec::<f32>()
    }

    const P: [[f32; 4]; 2] = [[0.5, 0.25, 0.25, 0.0], [0.0, 0.0, 0.5, 0.5]];

    #[test]
    fn a_reject_draws_from_the_residual() {
        let q = Some([0.25, 0.5, 0.0, 0.25]);
        assert_eq!(dist(P, q, 1, 0.0), [0.25, 0.0, 0.25, 0.0]);
    }

    #[test]
    fn a_one_hot_reject_zeroes_the_draft() {
        assert_eq!(dist(P, None, 0, 0.0), [0.0, 0.25, 0.25, 0.0]);
    }

    #[test]
    fn a_zero_residual_falls_back_to_the_target_row() {
        let q = Some([0.5, 0.25, 0.25, 0.0]);
        assert_eq!(dist(P, q, 3, 0.0), P[0]);
    }

    #[test]
    fn all_accepted_draws_from_the_bonus_row() {
        let q = Some([0.25, 0.5, 0.0, 0.25]);
        assert_eq!(dist(P, q, 1, 1.0), P[1]);
        assert_eq!(dist(P, None, 1, 1.0), P[1]);
    }
}

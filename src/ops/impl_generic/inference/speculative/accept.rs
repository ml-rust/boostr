//! The K accept tests of `speculative_verify`, on the device.

use super::draw::scalar_like;
use super::validate::VerifyShape;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, CumulativeOps, IndexingOps, ReduceOps, ScalarOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The draft ids of a call and the count of leading accepts.
pub struct LeadingAccepts<R: Runtime> {
    /// `[B, K]` F32 draft ids, cast from the I32 input.
    pub ids: Tensor<R>,
    /// `[B, 1]` F32 count `m` of leading accepted drafts.
    pub accepted: Tensor<R>,
}

/// Runs the K accept tests and counts the leading accepts. K must be above 0.
///
/// Draft `i` passes iff its id is in `[0, V)`, `q_i(d_i) > 0` and
/// `u_i * q_i(d_i) < p_i(d_i)`. `None` for `draft_probs` sets `q_i(d_i) = 1`.
/// `target` and `uniforms` must be contiguous.
pub fn leading_accepts<R: Runtime<DType = DType>>(
    client: &R::Client,
    target: &Tensor<R>,
    draft_probs: Option<&Tensor<R>>,
    draft_tokens: &Tensor<R>,
    uniforms: &Tensor<R>,
    shape: VerifyShape,
) -> Result<LeadingAccepts<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let VerifyShape {
        batch,
        drafts,
        vocab,
    } = shape;
    // I32 to F32 is exact for every id up to 2^24, and V is at most 2^24.
    // Rounding is monotone: an id at or above V stays there, a negative id stays negative.
    let ids = client.cast(draft_tokens, DType::F32)?;
    let non_negative = client.ge(&ids, &scalar_like(&ids, 0.0)?)?;
    let below_vocab = client.lt(&ids, &scalar_like(&ids, vocab as f64)?)?;
    let in_range = client.mul(&non_negative, &below_vocab)?;
    let index = client
        .cast(&client.mul(&ids, &in_range)?, DType::I32)?
        .reshape(&[batch, drafts, 1])?;
    let p_lead = target.narrow(1, 0, drafts)?.contiguous()?;
    let p_d = client
        .gather(&p_lead, 2, &index)?
        .reshape(&[batch, drafts])?;
    let q_d = match draft_probs {
        Some(q) => client
            .gather(&q.contiguous()?, 2, &index)?
            .reshape(&[batch, drafts])?,
        None => Tensor::<R>::ones(&[batch, drafts], DType::F32, target.device())?,
    };
    let u_lead = uniforms.narrow(1, 0, drafts)?.contiguous()?;
    let below = client.lt(&client.mul(&u_lead, &q_d)?, &p_d)?;
    let drawable = client.mul(&client.gt(&q_d, &scalar_like(&q_d, 0.0)?)?, &in_range)?;
    let accept = client.mul(&below, &drawable)?;
    let accepted = client.sum(&client.cumprod(&accept, 1)?, &[1], true)?;
    Ok(LeadingAccepts { ids, accepted })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    type T = Tensor<CpuRuntime>;

    /// One row, V = 2, K = 3. Returns `m`.
    fn count(p_d: [f32; 3], q_d: Option<[f32; 3]>, d: [i32; 3], u: [f32; 3]) -> f32 {
        let (client, device) = cpu_setup();
        let mut p = vec![0.0f32; 4 * 2];
        let mut q = vec![0.0f32; 3 * 2];
        for i in 0..3 {
            let col = d[i].clamp(0, 1) as usize;
            p[i * 2 + col] = p_d[i];
            q[i * 2 + col] = q_d.map_or(1.0, |q| q[i]);
        }
        let p = T::from_slice(&p, &[1, 4, 2], &device).expect("p");
        let q = T::from_slice(&q, &[1, 3, 2], &device).expect("q");
        let d = T::from_slice(&d, &[1, 3], &device).expect("d");
        let u = T::from_slice(&[u[0], u[1], u[2], 0.0], &[1, 4], &device).expect("u");
        let shape = VerifyShape {
            batch: 1,
            drafts: 3,
            vocab: 2,
        };
        let q = q_d.map(|_| &q);
        let lead = leading_accepts::<CpuRuntime>(&client, &p, q, &d, &u, shape).expect("accept");
        lead.accepted.to_vec::<f32>()[0]
    }

    #[test]
    fn the_scan_stops_at_the_first_reject() {
        let p = [0.5, 0.5, 0.5];
        assert_eq!(count(p, None, [0, 1, 0], [0.1, 0.2, 0.3]), 3.0);
        assert_eq!(count(p, None, [0, 1, 0], [0.1, 0.6, 0.3]), 1.0);
        assert_eq!(count(p, None, [0, 1, 0], [0.5, 0.1, 0.1]), 0.0);
    }

    #[test]
    fn the_test_scales_u_by_q_without_division() {
        let p = [0.2, 0.2, 0.2];
        let q = Some([0.4, 0.4, 0.4]);
        assert_eq!(count(p, q, [0, 0, 0], [0.49, 0.49, 0.49]), 3.0);
        assert_eq!(count(p, q, [0, 0, 0], [0.49, 0.51, 0.0]), 1.0);
    }

    #[test]
    fn a_draft_the_drafter_can_not_emit_is_rejected() {
        let q = Some([0.0, 0.5, 0.5]);
        assert_eq!(count([0.9, 0.9, 0.9], q, [0, 0, 0], [0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn an_out_of_range_draft_is_rejected() {
        let p = [0.9, 0.9, 0.9];
        assert_eq!(count(p, None, [0, 2, 0], [0.0, 0.0, 0.0]), 1.0);
        assert_eq!(count(p, None, [-1, 0, 0], [0.0, 0.0, 0.0]), 0.0);
    }
}

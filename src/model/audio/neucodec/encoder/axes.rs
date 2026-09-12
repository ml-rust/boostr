//! Rank-3 layout helpers: the channels-first/channels-last axis swap and the
//! time-axis truncation the two branches are cut with.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_narrow, var_permute};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Swap the last two axes of a rank-3 `Var`, materializing the result so the
/// downstream reshape/conv sees a contiguous buffer.
///
/// `[B, X, Y] <-> [B, Y, X]` — this is how the pipeline moves between the
/// channels-first layout the convs need and the channels-last layout `Linear`
/// and the quantizer need.
pub(super) fn to_time_last<R: Runtime<DType = DType>>(x: &Var<R>) -> Result<Var<R>> {
    if x.shape().len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected a rank-3 tensor, got {:?}", x.shape()),
        });
    }
    var_contiguous(&var_permute(x, &[0, 2, 1]).map_err(Error::Numr)?)
}

/// Length of the trailing time axis of a channels-first `[B, C, T]` `Var`.
pub(super) fn time_len<R: Runtime>(x: &Var<R>, arg: &'static str) -> Result<usize> {
    let shape = x.shape();
    match (shape.len(), shape.get(2).copied()) {
        (3, Some(t)) => Ok(t),
        _ => Err(Error::InvalidArgument {
            arg,
            reason: format!("expected channels-first [B, C, T], got {shape:?}"),
        }),
    }
}

/// `min(Ts, Ta)` over the two branches.
pub(super) fn min_time<R: Runtime>(semantic: &Var<R>, acoustic: &Var<R>) -> Result<usize> {
    let ts = time_len(semantic, "semantic")?;
    let ta = time_len(acoustic, "acoustic")?;
    Ok(ts.min(ta))
}

/// Keep the EARLIEST `len` frames of a channels-first `[B, C, T]` `Var`.
pub(super) fn narrow_time<R: Runtime<DType = DType>>(x: &Var<R>, len: usize) -> Result<Var<R>>
where
    R::Client: NeuCodecClient<R>,
{
    if time_len(x, "x")? == len {
        return Ok(x.alias());
    }
    var_contiguous(&var_narrow(x, 2, 0, len).map_err(Error::Numr)?)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Shared fixture: a non-tracked `Var` from host data.
    pub(in super::super) fn var(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            false,
        )
    }

    #[test]
    fn time_len_rejects_non_rank_3() {
        let (_client, device) = cpu_setup();
        let x = var(&[0.0; 6], &[2, 3], &device);
        assert!(time_len(&x, "x").is_err());
        assert!(to_time_last(&x).is_err());
    }

    /// Both branches are cut to `min(Ts, Ta)`, keeping the EARLIEST frames.
    #[test]
    fn branches_truncate_to_the_shorter_one() {
        let (_client, device) = cpu_setup();

        // [1, 2, 3] semantic (Ts = 3) and [1, 2, 5] acoustic (Ta = 5).
        let semantic = var(
            &(0..6).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 2, 3],
            &device,
        );
        let acoustic = var(
            &(0..10).map(|i| 100.0 + i as f32).collect::<Vec<_>>(),
            &[1, 2, 5],
            &device,
        );

        let min_len = min_time(&semantic, &acoustic).expect("min_time");
        assert_eq!(min_len, 3);

        let sem_cut = narrow_time(&semantic, min_len).expect("narrow semantic");
        let aco_cut = narrow_time(&acoustic, min_len).expect("narrow acoustic");
        assert_eq!(sem_cut.shape(), &[1, 2, 3]);
        assert_eq!(aco_cut.shape(), &[1, 2, 3]);

        // Earliest frames kept, tail dropped: rows are 100..103 and 105..108.
        let values = aco_cut
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>();
        assert_eq!(values, vec![100.0, 101.0, 102.0, 105.0, 106.0, 107.0]);
    }

    #[test]
    fn to_time_last_swaps_the_trailing_axes() {
        let (_client, device) = cpu_setup();
        let x = var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &device);
        let y = to_time_last(&x).expect("permute");
        assert_eq!(y.shape(), &[1, 3, 2]);
        assert_eq!(
            y.tensor().contiguous().expect("contiguous").to_vec::<f32>(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}

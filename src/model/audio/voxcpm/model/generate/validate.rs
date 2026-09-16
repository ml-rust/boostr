//! Shape validation and the stop-token decision, shared by every entry point
//! into the per-patch loop (`step`, `step_with_noise`, the capturing variant,
//! and `teacher_forced_conditioning`).

use super::*;

/// Did the stop classifier pick class 1? `logits` is `[1, 2]`. `argmax` runs
/// ON DEVICE and yields a single I64 index;
/// [`Tensor::item`](numr::tensor::Tensor::item) then copies THAT ONE value
/// back — 8 bytes per patch. The read is unavoidable (the answer drives
/// control flow) and deliberately the narrowest form: the logits never leave
/// the device, and nothing here calls `to_vec`.
pub(super) fn stop_predicted<R, C>(client: &C, logits: &Var<R>) -> Result<bool>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let shape = logits.shape();
    if shape.len() != 2 || shape[0] != 1 || shape[1] != 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected stop logits [1, 2], got {shape:?}"),
        });
    }
    let index = client
        .argmax(logits.tensor(), 1, false)
        .map_err(Error::Numr)?;
    Ok(index.item::<i64>().map_err(Error::Numr)? == STOP_CLASS)
}

/// Validate a `[1, hidden]` per-step hidden state, returning `hidden`.
pub(super) fn check_row<R: Runtime<DType = DType>>(arg: &'static str, v: &Var<R>) -> Result<usize> {
    let shape = v.shape();
    if shape.len() != 2 || shape[0] != 1 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected [1, hidden] (batch 1, one position), got {shape:?}"),
        });
    }
    Ok(shape[1])
}

/// Validate a rank-3 patch tensor against an exact expected shape.
pub(super) fn check_patch<R: Runtime<DType = DType>>(
    arg: &'static str,
    v: &Var<R>,
    expected: &[usize; 3],
) -> Result<()> {
    let shape = v.shape();
    if shape != expected.as_slice() {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected {expected:?}, got {shape:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::t;
    use crate::model::audio::voxcpm::minicpm4::model::tests::HIDDEN;
    use crate::test_utils::cpu_setup;

    /// The stop fixture must actually answer what it claims, in BOTH directions
    /// — every guard test below is vacuous otherwise.
    #[test]
    fn stop_chain_answers_its_configured_class() {
        let (client, device) = cpu_setup();
        for stop in [true, false] {
            let fx = fixture(stop, &device);
            let hidden = Var::new(t(&[1, HIDDEN], 0.4, &device), false);
            let logits = fx.aux.stop(&client, &hidden).expect("stop");
            assert_eq!(
                stop_predicted(&client, &logits).expect("argmax"),
                stop,
                "stop chain built for {stop} answered the other way"
            );
        }
    }
}

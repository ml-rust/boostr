//! Shape validation and the stop-token decision, shared by every entry point
//! into the per-patch loop (`step`, `step_with_noise`, the capturing variant,
//! and `teacher_forced_conditioning`).

use super::*;

/// Which rows' stop classifier picked class 1. `logits` is `[batch, 2]`.
/// `argmax` runs ON DEVICE and yields `[batch]` I64 indices; ONE `to_vec`
/// then copies those `batch` values back — eight bytes per row per patch.
/// The read is unavoidable (the answer drives control flow) and
/// deliberately the narrowest form: the logits never leave the device.
pub(super) fn stop_predicted<R, C>(client: &C, logits: &Var<R>, batch: usize) -> Result<Vec<bool>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let shape = logits.shape();
    if shape.len() != 2 || shape[0] != batch || shape[1] != 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected stop logits [{batch}, 2], got {shape:?}"),
        });
    }
    let index = client
        .argmax(logits.tensor(), 1, false)
        .map_err(Error::Numr)?;
    let classes: Vec<i64> = index.to_vec();
    if classes.len() != batch {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("argmax returned {} classes for {batch} rows", classes.len()),
        });
    }
    Ok(classes.into_iter().map(|c| c == STOP_CLASS).collect())
}

/// Validate a `[batch, hidden]` per-step hidden state, returning `hidden`.
pub(super) fn check_row<R: Runtime<DType = DType>>(
    arg: &'static str,
    v: &Var<R>,
    batch: usize,
) -> Result<usize> {
    let shape = v.shape();
    if shape.len() != 2 || shape[0] != batch {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected [{batch}, hidden] (one position per row), got {shape:?}"),
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
                stop_predicted(&client, &logits, 1).expect("argmax"),
                [stop],
                "stop chain built for {stop} answered the other way"
            );
        }
    }

    /// One read answers every row: a two-row logit tensor with one row per
    /// class comes back as two distinct answers.
    #[test]
    fn stop_predicted_answers_per_row() {
        let (client, device) = cpu_setup();
        let logits = Var::new(
            Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[2, 2], &device).expect("logits"),
            false,
        );
        assert_eq!(
            stop_predicted(&client, &logits, 2).expect("argmax"),
            [true, false]
        );
        assert!(stop_predicted(&client, &logits, 1).is_err());
    }
}

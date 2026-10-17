//! Key names of the optimizer state dictionary and the error constructor.

use crate::error::Error;

/// Key of the global step counter (AdamW and Lamb).
pub(crate) const STEP_KEY: &str = "__step";
/// Field of the F32 master copy of a narrow parameter.
pub(crate) const MASTER: &str = "master";
/// Field of the first moment (AdamW and Lamb).
pub(crate) const EXP_AVG: &str = "exp_avg";
/// Field of the second moment (AdamW and Lamb).
pub(crate) const EXP_AVG_SQ: &str = "exp_avg_sq";
/// Field of the squared-gradient accumulator (AdaGrad).
pub(crate) const SUM_SQ: &str = "sum_sq";
/// Field of the velocity buffer (Sgd).
pub(crate) const MOMENTUM_BUFFER: &str = "momentum_buffer";

/// Build the error every state dictionary operation returns.
pub(crate) fn fail(reason: String) -> Error {
    Error::TrainingError { reason }
}

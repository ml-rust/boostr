//! Input checks of [`MaybeRotatedLinear::row_logits_f32`], and
//! [`name_out_of_range`], which names the id behind a failed device-side
//! bounds check.
//!
//! [`MaybeRotatedLinear::row_logits_f32`]: super::layer::MaybeRotatedLinear::row_logits_f32

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

/// `(Q, K)` once `input` is `[Q, in_features]` float and `rows` is
/// `[Q, K]` i64 on the same device with `Q, K >= 1`.
pub(super) fn check_row_inputs<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    rows: &Tensor<R>,
    in_features: usize,
) -> Result<(usize, usize)> {
    let bad_input = |reason: String| Error::InvalidArgument {
        arg: "input",
        reason,
    };
    let bad_rows = |reason: String| Error::InvalidArgument {
        arg: "rows",
        reason,
    };
    let in_shape = input.shape();
    if in_shape.len() != 2 || in_shape[0] == 0 || in_shape[1] != in_features {
        return Err(bad_input(format!(
            "must be [Q >= 1, {in_features}], got {in_shape:?}"
        )));
    }
    if !input.dtype().is_float() {
        return Err(bad_input(format!(
            "must be a float dtype, got {:?}",
            input.dtype()
        )));
    }
    let q = in_shape[0];
    let rows_shape = rows.shape();
    if rows_shape.len() != 2 || rows_shape[0] != q || rows_shape[1] == 0 {
        return Err(bad_rows(format!(
            "must be [{q}, K >= 1] to match input, got {rows_shape:?}"
        )));
    }
    if rows.dtype() != DType::I64 {
        return Err(bad_rows(format!("must be I64, got {:?}", rows.dtype())));
    }
    if !rows.device().is_same(input.device()) {
        return Err(bad_rows(format!(
            "is on {}, input on {}",
            rows.device().name(),
            input.device().name()
        )));
    }
    Ok((q, rows_shape[1]))
}

/// Turns an `index_select` out-of-bounds error over `ids` into an
/// [`Error::InvalidArgument`] naming the first id outside `[0, bound)`
/// and its flat position. Every other error passes through unchanged.
///
/// The device check does not report the offending id on every backend,
/// so this error path reads `ids` back once to name it. The success path
/// reads nothing.
pub(crate) fn name_out_of_range<R: Runtime<DType = DType>>(
    err: Error,
    ids: &Tensor<R>,
    bound: usize,
    arg: &'static str,
    what: &str,
) -> Error {
    if !matches!(
        err,
        Error::Numr(numr::error::Error::IndexOutOfBounds { .. })
    ) {
        return err;
    }
    let Ok(values) = ids.contiguous().and_then(|t| t.try_to_vec::<i64>()) else {
        return err;
    };
    let first_bad = values
        .iter()
        .enumerate()
        .find(|(_, v)| usize::try_from(**v).ok().is_none_or(|v| v >= bound));
    match first_bad {
        Some((pos, value)) => Error::InvalidArgument {
            arg,
            reason: format!("{what} {value} at flat position {pos} is outside [0, {bound})"),
        },
        None => err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const IN: usize = 8;

    fn input(device: &CpuDevice, shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::ones(shape, DType::F32, device).unwrap()
    }

    fn rows(device: &CpuDevice, ids: &[i64], q: usize) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(ids, &[q, ids.len() / q], device).unwrap()
    }

    #[test]
    fn accepts_matching_inputs() {
        let (_client, device) = cpu_setup();
        let got = check_row_inputs(&input(&device, &[2, IN]), &rows(&device, &[1; 6], 2), IN);
        assert_eq!(got.unwrap(), (2, 3));
    }

    #[test]
    fn rejects_each_contract_break() {
        let (_client, device) = cpu_setup();
        let x = input(&device, &[2, IN]);
        let ok = rows(&device, &[1; 6], 2);
        let refused = |input: &Tensor<CpuRuntime>, rows: &Tensor<CpuRuntime>| {
            check_row_inputs(input, rows, IN).unwrap_err().to_string()
        };

        assert!(refused(&input(&device, &[2, IN - 1]), &ok).contains("Q >= 1"));
        assert!(refused(&input(&device, &[0, IN]), &ok).contains("Q >= 1"));
        let ints = Tensor::<CpuRuntime>::zeros(&[2, IN], DType::I64, &device).unwrap();
        assert!(refused(&ints, &ok).contains("float"));
        assert!(refused(&x, &rows(&device, &[1; 6], 3)).contains("K >= 1"));
        let empty = Tensor::<CpuRuntime>::zeros(&[2, 0], DType::I64, &device).unwrap();
        assert!(refused(&x, &empty).contains("K >= 1"));
        let as_i32 = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::I32, &device).unwrap();
        assert!(refused(&x, &as_i32).contains("I64"));
    }

    #[test]
    fn names_the_first_id_out_of_range() {
        let (_client, device) = cpu_setup();
        let ids = rows(&device, &[3, 9, -1, 12], 1);
        let bounds_err = Error::Numr(numr::error::Error::IndexOutOfBounds { index: 0, size: 9 });
        let err = name_out_of_range(bounds_err, &ids, 9, "rows", "row id").to_string();
        assert!(
            err.contains("row id 9 at flat position 1 is outside [0, 9)"),
            "{err}"
        );

        let other = Error::ModelError {
            reason: "unrelated".into(),
        };
        let err = name_out_of_range(other, &ids, 9, "rows", "row id").to_string();
        assert!(err.contains("unrelated"), "{err}");
    }
}

//! Input checks for the CPU varlen kernels: dtype, offset dtype, and head layout.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Shared dtype of the float inputs: F32, F16 or BF16.
///
/// The kernels compute in F32. A half-precision input is cast up first, and
/// the results are cast back to this dtype.
pub(super) fn float_dtype(
    op: &'static str,
    tensors: &[(&'static str, &Tensor<CpuRuntime>)],
) -> Result<DType> {
    let Some(&(_, first)) = tensors.first() else {
        return Err(Error::InvalidArgument {
            arg: "inputs",
            reason: format!("{op} (CPU): no float inputs to check"),
        });
    };
    let dtype = first.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "{op} (CPU): supports F32, F16 and BF16, got {dtype:?}; cast the inputs first"
            ),
        });
    }
    for &(name, t) in tensors {
        if t.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "{op} (CPU): {name} is {:?}, but the inputs are {dtype:?}; \
                     every float input must share one dtype",
                    t.dtype()
                ),
            });
        }
    }
    Ok(dtype)
}

/// Check that an offset table is an I32 tensor.
pub(super) fn offsets_i32(
    op: &'static str,
    arg: &'static str,
    t: &Tensor<CpuRuntime>,
) -> Result<()> {
    if t.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("{op} (CPU): {arg} must be I32, got {:?}", t.dtype()),
        });
    }
    Ok(())
}

/// Check the head counts, so the GQA mapping `h / (H / H_kv)` is defined.
pub(super) fn head_counts(op: &'static str, num_heads: usize, num_kv_heads: usize) -> Result<()> {
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "{op} (CPU): num_heads {num_heads} must be a non-zero multiple of \
                 num_kv_heads {num_kv_heads}"
            ),
        });
    }
    Ok(())
}

/// Check that a packed tensor is `[tokens, heads, head_dim]` with the given heads.
pub(super) fn packed_shape(
    op: &'static str,
    arg: &'static str,
    t: &Tensor<CpuRuntime>,
    heads: usize,
    head_dim: usize,
) -> Result<usize> {
    let shape = t.shape();
    if shape.len() != 3 || shape[1] != heads || shape[2] != head_dim {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op} (CPU): {arg} must be [tokens, {heads}, {head_dim}], got {shape:?}"
            ),
        });
    }
    Ok(shape[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn float_dtype_rejects_f64_and_mixed() {
        let (_, dev) = cpu_setup();
        let a = Tensor::<CpuRuntime>::zeros(&[2, 1, 4], DType::F64, &dev).unwrap();
        let b = Tensor::<CpuRuntime>::zeros(&[2, 1, 4], DType::F32, &dev).unwrap();
        let e = float_dtype("op", &[("q", &a)]).err().unwrap();
        assert!(e.to_string().contains("supports F32, F16 and BF16"), "{e}");
        let e = float_dtype("op", &[("q", &b), ("k", &a)]).err().unwrap();
        assert!(e.to_string().contains("share one dtype"), "{e}");
        assert_eq!(float_dtype("op", &[("q", &b)]).unwrap(), DType::F32);
    }

    #[test]
    fn head_counts_rejects_zero_and_non_multiple() {
        assert!(head_counts("op", 8, 0).is_err());
        assert!(head_counts("op", 8, 3).is_err());
        assert!(head_counts("op", 8, 2).is_ok());
    }
}

//! Half-precision support for `GrammarDfaOps`, shared by every backend.
//!
//! The mask kernels read F32 only. F16 and BF16 logits cast to F32 on the device,
//! take the F32 kernel, and cast back. Both casts are exact for the unmasked values.

use crate::error::{Error, Result};
use crate::inference::grammar::device::check_logits;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Whether a backend casts half-precision logits without a host transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfLogits {
    /// The backend casts F16 and BF16 on the device.
    Cast,
    /// The backend has F32 only. F16 and BF16 are an error.
    Refuse,
}

/// Mask `logits` with `mask_f32` and return a tensor of the input dtype.
///
/// `mask_f32` receives F32 logits and returns the masked F32 tensor.
/// An F32 input goes straight to `mask_f32` with no cast.
/// Any dtype other than F32, F16 and BF16 is an error that names the dtype.
pub fn grammar_mask_any_float_impl<R, C>(
    client: &C,
    logits: &Tensor<R>,
    vocab_size: usize,
    half: HalfLogits,
    mask_f32: impl FnOnce(&Tensor<R>) -> Result<Tensor<R>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    check_logits(logits, vocab_size)?;
    let dtype = logits.dtype();
    if dtype == DType::F32 {
        return mask_f32(logits);
    }
    if half == HalfLogits::Refuse {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "grammar_dfa_mask_logits: this backend masks F32 logits only, got {dtype:?}. \
                 Cast the logits to F32 on the device first"
            ),
        });
    }
    let wide = client.cast(logits, DType::F32)?;
    let masked = mask_f32(&wide)?;
    Ok(client.cast(&masked, dtype)?)
}

#[cfg(all(test, feature = "f16"))]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Every non-NaN 16-bit pattern survives a cast to F32 and back with the same bits.
    fn round_trip<T: numr::dtype::Element + Copy>(
        from_bits: impl Fn(u16) -> T,
        to_bits: impl Fn(T) -> u16,
        is_nan: impl Fn(T) -> bool,
        dtype: DType,
    ) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let values: Vec<T> = (0..=u16::MAX)
            .map(&from_bits)
            .filter(|&v| !is_nan(v))
            .collect();
        let t = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], &device).unwrap();
        let wide = client.cast(&t, DType::F32).unwrap();
        let back = client.cast(&wide, dtype).unwrap();
        assert_eq!(back.dtype(), dtype);
        let got: Vec<u16> = back.to_vec::<T>().into_iter().map(&to_bits).collect();
        let want: Vec<u16> = values.into_iter().map(&to_bits).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn f16_round_trip_through_f32_is_bit_exact() {
        round_trip(
            half::f16::from_bits,
            half::f16::to_bits,
            half::f16::is_nan,
            DType::F16,
        );
    }

    #[test]
    fn bf16_round_trip_through_f32_is_bit_exact() {
        round_trip(
            half::bf16::from_bits,
            half::bf16::to_bits,
            half::bf16::is_nan,
            DType::BF16,
        );
    }
}

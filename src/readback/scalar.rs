//! Host readback of single-element tensors.
//!
//! Every scalar a trainer transfers to the host — a loss, a gradient norm, a
//! mask count — carries the tensor's own dtype, which under mixed precision is
//! BF16 or F16 rather than F32. These helpers are the only correct way to read
//! such a scalar.

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Read a single-element tensor as `f32`, casting to F32 on device first.
///
/// `Tensor::item::<f32>` is a raw byte reinterpretation, not a conversion: it
/// sizes the copy from `size_of::<f32>()` and never consults the tensor's
/// dtype. Against a BF16 or F16 scalar it asks the buffer for 4 bytes where 2
/// were allocated. The low half is the real value and the high half — the whole
/// f32 sign and exponent field — is uninitialized memory, because `Tensor` does
/// not zero its allocations. The result is a plausible-looking wrong number,
/// NaN, or a denormal, never an error. Against F64 it reads half the bytes.
///
/// Casting to F32 first makes the readback exact for every float dtype. For an
/// F32 tensor the cast is a no-op and the value is bit-identical to reading it
/// directly, so this is safe to apply unconditionally.
pub fn scalar_f32<R, C>(client: &C, tensor: &Tensor<R>) -> Result<f32>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    Ok(client.cast(tensor, DType::F32)?.item::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// F32 is the pre-existing path: the cast must not perturb the value.
    #[test]
    fn scalar_f32_is_exact_for_f32() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::from_slice(&[15.5f32], &[1], &device).unwrap();
        assert_eq!(scalar_f32(&client, &t).expect("readback runs"), 15.5);
    }

    /// The narrow dtypes are where `item::<f32>` over-reads the buffer. 15.5 is
    /// exactly representable in BF16 and F16, so the expected value is exact and
    /// any byte-reinterpretation result fails the comparison. BF16 and F16
    /// arithmetic lives behind numr's `f16` feature, so this needs
    /// `--features f16`; the F64 case below covers the same defect ungated.
    #[cfg(feature = "f16")]
    #[test]
    fn scalar_f32_reads_narrow_dtypes_at_their_own_width() {
        let (client, device) = cpu_setup();
        let f32_t = Tensor::<CpuRuntime>::from_slice(&[15.5f32], &[1], &device).unwrap();

        for dtype in [DType::BF16, DType::F16] {
            let narrow = client.cast(&f32_t, dtype).expect("cast runs");
            assert_eq!(narrow.dtype(), dtype);
            let value = scalar_f32(&client, &narrow).expect("readback runs");
            assert_eq!(value, 15.5, "{dtype:?} read back as {value}");
        }
    }

    /// F64 is the opposite error: `item::<f32>` reads half the bytes.
    #[test]
    fn scalar_f32_reads_f64_at_its_own_width() {
        let (client, device) = cpu_setup();
        let f32_t = Tensor::<CpuRuntime>::from_slice(&[15.5f32], &[1], &device).unwrap();
        let wide = client.cast(&f32_t, DType::F64).expect("cast runs");
        assert_eq!(scalar_f32(&client, &wide).expect("readback runs"), 15.5);
    }
}

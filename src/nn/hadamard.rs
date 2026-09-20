//! Hadamard activation-rotation basis, matching llama.cpp's
//! forward/inverse contract (see `crate::format::gguf::hadamard_contract`).
//!
//! Forward (before a rotated matmul): sign-flip the activation, then
//! transform. Inverse (after a rotated lookup): transform, then sign-flip.
//! The two orders are NOT interchangeable, so each gets its own method.

use crate::error::{Error, Result};
use crate::quant::traits::Rotation;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One rotation basis: block width and an optional +/-1 sign vector in the
/// activation dtype.
///
/// `Clone` is cheap: `Tensor::clone` shares storage, it does not copy device
/// memory.
pub struct HadamardRotation<R: Runtime> {
    block_size: usize,
    signs: Option<Tensor<R>>,
}

impl<R: Runtime> Clone for HadamardRotation<R> {
    fn clone(&self) -> Self {
        Self {
            block_size: self.block_size,
            signs: self.signs.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> HadamardRotation<R> {
    /// Builds the rotation. `signs` comes from the GGUF contract as `i8`
    /// (`+1`/`-1`) and is materialized as a 1-D tensor of `dtype` on
    /// `device` — the same dtype the activation this rotates will carry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `dtype` is not one of the
    /// floating-point dtypes `fwht` accepts (`F64`, `F32`, `F16`, `BF16`).
    pub fn new(
        block_size: usize,
        signs: Option<&[i8]>,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Self> {
        let signs = match signs {
            None => None,
            Some(values) => Some(signs_tensor::<R>(values, dtype, device)?),
        };
        Ok(Self { block_size, signs })
    }

    /// The block width the transform is applied per segment of.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The sign vector's width, when this rotation carries one.
    pub fn width(&self) -> Option<usize> {
        self.signs.as_ref().map(|s| s.shape()[0])
    }

    /// Identity for "is this the same rotation" checks: the signs tensor's
    /// storage pointer, `None` in identity mode (no signs).
    ///
    /// Not `Tensor::id()`: `Attach::rotation`
    /// (`crate::model::qwen35::model::gguf`) caches one materialized signs
    /// tensor per width and hands out `Tensor::clone`s of it — a clone
    /// shares storage but gets a fresh `TensorId`, so `Tensor::id()` would
    /// call two clones of the same rotation "different".
    pub fn signs_ptr(&self) -> Option<u64> {
        self.signs.as_ref().map(|s| s.storage().ptr())
    }

    /// `true` when `self` and `other` transform with the same block width
    /// over the same underlying sign storage (both `None` counts as equal:
    /// two identity rotations of the same width behave identically).
    pub fn same_rotation_as(&self, other: &Self) -> bool {
        self.block_size == other.block_size && self.signs_ptr() == other.signs_ptr()
    }

    /// This rotation as the argument a quantized matmul folds into its
    /// activation quantization (`QuantMatmulOps::quant_matmul_batch_rotated`)
    /// for the activation `x`: the same operation as [`Self::forward`] on
    /// `x`, with the same dtype check.
    pub fn rotation_for(&self, x: &Tensor<R>) -> Result<Rotation<'_, R>> {
        if let Some(signs) = &self.signs {
            check_dtype_match(x, signs)?;
        }
        Ok(Rotation {
            block_size: self.block_size,
            signs: self.signs.as_ref(),
        })
    }

    /// Forward rotation: sign-multiply then transform. Used before a
    /// rotated matmul. `fwht` multiplies `signs` in before the transform in
    /// one call, so this is a single kernel launch.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: FwhtOps<R>,
    {
        if let Some(signs) = &self.signs {
            check_dtype_match(x, signs)?;
        }
        client
            .fwht(x, self.block_size, self.signs.as_ref())
            .map_err(Error::Numr)
    }

    /// Inverse rotation: transform then sign-multiply. Used after a rotated
    /// lookup (e.g. `token_embd.weight`).
    pub fn inverse<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: FwhtOps<R> + BinaryOps<R>,
    {
        let transformed = client.fwht(x, self.block_size, None).map_err(Error::Numr)?;
        match &self.signs {
            None => Ok(transformed),
            Some(signs) => {
                check_dtype_match(&transformed, signs)?;
                client.mul(&transformed, signs).map_err(Error::Numr)
            }
        }
    }
}

fn check_dtype_match<R: Runtime<DType = DType>>(x: &Tensor<R>, signs: &Tensor<R>) -> Result<()> {
    if x.dtype() != signs.dtype() {
        return Err(Error::DTypeMismatch {
            expected: x.dtype(),
            got: signs.dtype(),
        });
    }
    Ok(())
}

/// Converts `+1`/`-1` `i8` signs into a 1-D tensor of `dtype` on `device`.
/// Host-side conversion — no client/runtime op exists to cast a freshly
/// materialized host array, and every listed dtype has a direct `f64`
/// (or narrower) constructor.
fn signs_tensor<R: Runtime<DType = DType>>(
    values: &[i8],
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let shape = [values.len()];
    match dtype {
        DType::F64 => build_signs::<R, f64>(values, &shape, device, |v| v as f64),
        DType::F32 => build_signs::<R, f32>(values, &shape, device, |v| v as f32),
        // `half::f16`/`half::bf16` implement numr's `Element` only under
        // numr's own `f16` feature (`boostr`'s `f16` feature enables it) —
        // without it these two arms fail to type-check, so they compile only
        // then; the `other` arm below still covers F16/BF16 when the feature
        // is off, refusing them explicitly instead of failing to build.
        #[cfg(feature = "f16")]
        DType::F16 => {
            build_signs::<R, half::f16>(values, &shape, device, |v| half::f16::from_f32(v as f32))
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            build_signs::<R, half::bf16>(values, &shape, device, |v| half::bf16::from_f32(v as f32))
        }
        other => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("Hadamard signs need a floating-point activation dtype, got {other:?}"),
        }),
    }
}

/// Shared conversion path for every dtype arm above: one host `Vec<T>`
/// built by `convert`, then one `Tensor::from_slice` call.
fn build_signs<R: Runtime<DType = DType>, T: numr::dtype::Element>(
    values: &[i8],
    shape: &[usize],
    device: &R::Device,
    convert: impl Fn(i8) -> T,
) -> Result<Tensor<R>> {
    let data: Vec<T> = values.iter().map(|&v| convert(v)).collect();
    Tensor::<R>::from_slice(&data, shape, device).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Explicit Sylvester-ordered Hadamard matrix `H[i][j] = (-1)^popcount(i&j)/sqrt(n)`.
    fn sylvester_hadamard(n: usize) -> Vec<f32> {
        let scale = 1.0f32 / (n as f32).sqrt();
        let mut h = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let bit = ((i & j).count_ones() % 2) as i32;
                h[i * n + j] = if bit == 0 { scale } else { -scale };
            }
        }
        h
    }

    /// Reference forward: for each row of `x` and each `block_size` segment,
    /// `(x * signs) @ H^T` (H is symmetric, so `@ H` works too).
    fn reference_forward(
        x: &[f32],
        rows: usize,
        width: usize,
        block: usize,
        signs: &[f32],
    ) -> Vec<f32> {
        let h = sylvester_hadamard(block);
        let mut out = vec![0.0f32; rows * width];
        for r in 0..rows {
            for seg in (0..width).step_by(block) {
                for i in 0..block {
                    let mut acc = 0.0f32;
                    for j in 0..block {
                        acc += (x[r * width + seg + j] * signs[seg + j]) * h[i * block + j];
                    }
                    out[r * width + seg + i] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn forward_matches_hand_computed_sylvester_transform() {
        let (client, device) = cpu_setup();
        let rows = 2;
        let width = 16;
        let block = 8;
        let x_data: Vec<f32> = (0..rows * width).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let signs_i8: Vec<i8> = (0..width)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();
        let signs_f32: Vec<f32> = signs_i8.iter().map(|&v| v as f32).collect();

        let rotation =
            HadamardRotation::<CpuRuntime>::new(block, Some(&signs_i8), DType::F32, &device)
                .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[rows, width], &device).unwrap();
        let out = rotation.forward(&client, &x).unwrap();
        let got: Vec<f32> = out.to_vec();

        let expected = reference_forward(&x_data, rows, width, block, &signs_f32);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {g}, expected {e}");
        }
    }

    #[test]
    fn inverse_and_forward_round_trip() {
        let (client, device) = cpu_setup();
        let rows = 2;
        let width = 16;
        let block = 8;
        let x_data: Vec<f32> = (0..rows * width).map(|i| (i as f32) * 0.07 + 0.2).collect();
        let signs_i8: Vec<i8> = (0..width)
            .map(|i| if (i / 3) % 2 == 0 { 1 } else { -1 })
            .collect();

        let rotation =
            HadamardRotation::<CpuRuntime>::new(block, Some(&signs_i8), DType::F32, &device)
                .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[rows, width], &device).unwrap();

        let forward_then_inverse = rotation
            .inverse(&client, &rotation.forward(&client, &x).unwrap())
            .unwrap();
        let fi: Vec<f32> = forward_then_inverse.to_vec();
        for (g, e) in fi.iter().zip(x_data.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "inverse(forward(x)) got {g}, expected {e}"
            );
        }

        let inverse_then_forward = rotation
            .forward(&client, &rotation.inverse(&client, &x).unwrap())
            .unwrap();
        let ifr: Vec<f32> = inverse_then_forward.to_vec();
        for (g, e) in ifr.iter().zip(x_data.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "forward(inverse(x)) got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn same_rotation_as_matches_clones_and_rejects_different_signs() {
        let (_client, device) = cpu_setup();
        let signs_a: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let signs_b: Vec<i8> = vec![-1, 1, -1, 1, -1, 1, -1, 1];

        let rotation_a =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs_a), DType::F32, &device).unwrap();
        let rotation_a_clone = rotation_a.clone();
        let rotation_b =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs_b), DType::F32, &device).unwrap();
        let identity_a = HadamardRotation::<CpuRuntime>::new(8, None, DType::F32, &device).unwrap();
        let identity_b = HadamardRotation::<CpuRuntime>::new(8, None, DType::F32, &device).unwrap();

        assert!(rotation_a.same_rotation_as(&rotation_a_clone));
        assert!(!rotation_a.same_rotation_as(&rotation_b));
        assert!(identity_a.same_rotation_as(&identity_b));
        assert!(!rotation_a.same_rotation_as(&identity_a));
        assert_eq!(rotation_a.signs_ptr(), rotation_a_clone.signs_ptr());
        assert!(identity_a.signs_ptr().is_none());
    }

    #[test]
    fn dtype_mismatch_between_x_and_signs_errors() {
        let (client, device) = cpu_setup();
        let signs_i8: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        // Signs materialized as F32, but `x` is F64 — must not cast silently.
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs_i8), DType::F32, &device).unwrap();

        let x_data = [1.0f64; 8];
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 8], &device).unwrap();

        assert!(rotation.forward(&client, &x).is_err());
        assert!(rotation.inverse(&client, &x).is_err());
    }
}

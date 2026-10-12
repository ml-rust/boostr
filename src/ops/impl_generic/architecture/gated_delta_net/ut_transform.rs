//! UT transform of the chunked gated delta rule.

use crate::error::Result;
use numr::algorithm::linalg::{LinearAlgebraAlgorithms, linalg_demote, linalg_promote};
use numr::dtype::DType;
use numr::ops::{ShapeOps, TypeConversionOps, UtilityOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `T = (I + A)^-1` for strictly lower-triangular `A: [N, n_chunks, cs, cs]`.
///
/// Forward substitution solves `(I + A) T = I` for every `(N, chunk)`
/// matrix in one batched numr call over `[N * n_chunks, cs, cs]`.
/// Substitution is backward stable. A product of matrix powers is not: the
/// powers of `A` grow large when keys overlap, and their cancellation loses
/// F32 precision.
///
/// With `unit_diagonal`, the solve reads only the strict lower triangle of
/// its matrix and takes the diagonal as 1. `A` therefore stands for `I + A`
/// with no extra add.
///
/// F16 and BF16 solve in F32 and return in the input dtype. F32 and F64
/// solve in their own dtype.
pub fn ut_transform<R, C>(
    client: &C,
    a: &Tensor<R>,
    n: usize,
    n_chunks: usize,
    cs: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + ShapeOps<R> + TypeConversionOps<R> + UtilityOps<R>,
{
    let batch = n * n_chunks;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let a = a.contiguous()?.reshape(&[batch, cs, cs])?;
    let eye = client.eye(cs, None, a.dtype())?.reshape(&[1, cs, cs])?;
    let eye = client.repeat(&eye, &[batch, 1, 1])?;
    let t = client.solve_triangular_lower(&a, &eye, true)?;
    let t = linalg_demote(client, t, original_dtype)?;
    Ok(t.reshape(&[n, n_chunks, cs, cs])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::{BinaryOps, MatmulOps};
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    /// Largest `|(I + A) T - I|` over every matrix in `[n, n_chunks, cs, cs]`.
    fn ut_residual(client: &CpuClient, a: &[f32], n: usize, n_chunks: usize, cs: usize) -> f32 {
        let device = client.device();
        let a = Tensor::<CpuRuntime>::from_slice(a, &[n, n_chunks, cs, cs], device).unwrap();
        let t = ut_transform(client, &a, n, n_chunks, cs).unwrap();
        assert_eq!(t.shape(), &[n, n_chunks, cs, cs]);
        let eye = client
            .eye(cs, None, DType::F32)
            .unwrap()
            .reshape(&[1, 1, cs, cs])
            .unwrap();
        let lhs = client.add(&eye, &a).unwrap();
        let prod = client.matmul(&lhs, &t).unwrap();
        let resid = client.sub(&prod, &eye).unwrap();
        resid
            .to_vec::<f32>()
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn inverts_small_batch() {
        let (client, _) = cpu_setup();
        // Two strictly lower 3x3 matrices in a [1, 2, 3, 3] layout.
        let a = [
            0.0f32, 0.0, 0.0, 0.5, 0.0, 0.0, -1.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 3.0, 0.75, 0.0,
        ];
        let resid = ut_residual(&client, &a, 1, 2, 3);
        assert!(resid < 1e-6, "residual {resid}");
    }

    /// Only the strict lower triangle of `A` is read: the solve takes `I + A`.
    #[test]
    fn ignores_diagonal_of_a() {
        let (client, device) = cpu_setup();
        let strict = [0.0f32, 0.0, 0.5, 0.0];
        let dirty = [9.0f32, 3.0, 0.5, -4.0];
        let run = |data: &[f32]| {
            let a = Tensor::<CpuRuntime>::from_slice(data, &[1, 1, 2, 2], &device).unwrap();
            ut_transform(&client, &a, 1, 1, 2).unwrap().to_vec::<f32>()
        };
        assert_eq!(run(&strict), run(&dirty));
    }

    /// Strictly lower entries near 0.8 over 64 x 64 chunks: the case of
    /// nearly coincident keys. Forward substitution is backward stable, so
    /// the residual stays near the F32 unit roundoff times the row sums of
    /// `|I + A| |T|`, far below 1e-4.
    #[test]
    fn overlapping_keys_residual() {
        let (client, _) = cpu_setup();
        let (n, n_chunks, cs) = (2, 2, 64);
        let mut a = vec![0.0f32; n * n_chunks * cs * cs];
        for m in 0..n * n_chunks {
            for i in 0..cs {
                for j in 0..i {
                    let phase = (m * cs * cs + i * cs + j) as f32 * 0.37;
                    a[m * cs * cs + i * cs + j] = 0.8 + 0.05 * phase.sin();
                }
            }
        }
        let resid = ut_residual(&client, &a, n, n_chunks, cs);
        assert!(resid < 1e-4, "residual {resid}");
    }
}

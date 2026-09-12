//! Mamba3 complex RoPE on the B/C projections.

use super::layer::Mamba3;
use crate::error::{Error, Result};
use crate::model::mamba::ssm::var_contiguous;
use numr::autograd::{
    Var, var_add, var_cat, var_cos, var_mul, var_narrow, var_reshape, var_sin, var_sub,
};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba3<R> {
    pub(in crate::model::mamba::mamba3) fn apply_rope<C>(
        &self,
        client: &C,
        input: &Var<R>,
        angles: &Var<R>,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let nheads = shape[2];
        let d_state = shape[3];
        if !d_state.is_multiple_of(2) {
            return Err(Error::ModelError {
                reason: format!("complex RoPE requires even state dimension, got {d_state}"),
            });
        }
        let half = d_state / 2;
        let pairs = var_reshape(input, &[batch, seq_len, nheads, half, 2]).map_err(Error::Numr)?;
        let real_pair = var_contiguous(&var_narrow(&pairs, 4, 0, 1).map_err(Error::Numr)?)?;
        let real = var_reshape(&real_pair, &[batch, seq_len, nheads, half]).map_err(Error::Numr)?;
        let imag_pair = var_contiguous(&var_narrow(&pairs, 4, 1, 1).map_err(Error::Numr)?)?;
        let imag = var_reshape(&imag_pair, &[batch, seq_len, nheads, half]).map_err(Error::Numr)?;
        let cos = var_cos(angles, client).map_err(Error::Numr)?;
        let sin = var_sin(angles, client).map_err(Error::Numr)?;

        let real_cos = var_mul(&real, &cos, client).map_err(Error::Numr)?;
        let imag_sin = var_mul(&imag, &sin, client).map_err(Error::Numr)?;
        let real_new = var_sub(&real_cos, &imag_sin, client).map_err(Error::Numr)?;

        let real_sin = var_mul(&real, &sin, client).map_err(Error::Numr)?;
        let imag_cos = var_mul(&imag, &cos, client).map_err(Error::Numr)?;
        let imag_new = var_add(&real_sin, &imag_cos, client).map_err(Error::Numr)?;

        let real_new =
            var_reshape(&real_new, &[batch, seq_len, nheads, half, 1]).map_err(Error::Numr)?;
        let imag_new =
            var_reshape(&imag_new, &[batch, seq_len, nheads, half, 1]).map_err(Error::Numr)?;
        let rotated = var_cat(&[&real_new, &imag_new], 4, client).map_err(Error::Numr)?;
        var_reshape(&rotated, &[batch, seq_len, nheads, d_state]).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny_mamba3;
    use super::*;
    use crate::model::mamba::mamba3::config::Mamba3Config;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_complex_rope_matches_f64_reference() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(2)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(4)
            .with_complex_rope(true)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);
        let mamba = tiny_mamba3(config);

        let tensor_data = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0];
        let angles_data = [
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_2,
        ];
        let tensor = Var::new(
            Tensor::<CpuRuntime>::from_slice(&tensor_data, &[1, 2, 1, 4], &device).unwrap(),
            false,
        );
        let angles = Var::new(
            Tensor::<CpuRuntime>::from_slice(&angles_data, &[1, 2, 1, 2], &device).unwrap(),
            false,
        );

        let out = mamba.apply_rope(&client, &tensor, &angles).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        let expected = rope_reference(&tensor_data, &angles_data);

        for (i, (actual, expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 2e-5,
                "idx={i}: actual={actual}, expected={expected}"
            );
        }
    }

    fn rope_reference(tensor: &[f32; 8], angles: &[f32; 4]) -> Vec<f64> {
        let mut out = Vec::with_capacity(8);
        for t in 0..2 {
            for pair in 0..2 {
                let base = t * 4 + pair * 2;
                let real = tensor[base] as f64;
                let imag = tensor[base + 1] as f64;
                let angle = angles[t * 2 + pair] as f64;
                out.push(real * angle.cos() - imag * angle.sin());
                out.push(real * angle.sin() + imag * angle.cos());
            }
        }
        out
    }
}

//! Mamba3 MIMO up/down projections around the SSM scan.

use super::layer::Mamba3;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime> Mamba3<R> {
    pub(in crate::model::mamba::mamba3) fn apply_mimo_up<C>(
        &self,
        client: &C,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        if self.config.mimo_rank == 0 {
            // `alias`, not `clone` — `Var::clone` mints a fresh TensorId.
            return Ok(input.alias());
        }
        let up_proj = self.mimo_x_up.as_ref().ok_or_else(|| Error::ModelError {
            reason: "Mamba3 mimo_rank > 0 requires mimo_x_up".into(),
        })?;
        let flat = var_reshape(
            input,
            &[batch * seq_len * self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)?;
        let up = up_proj.forward(client, &flat)?;
        var_reshape(
            &up,
            &[
                batch,
                seq_len,
                self.config.nheads,
                self.config.headdim * self.config.mimo_rank,
            ],
        )
        .map_err(Error::Numr)
    }

    pub(in crate::model::mamba::mamba3) fn apply_mimo_down<C>(
        &self,
        client: &C,
        input: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        if self.config.mimo_rank == 0 {
            // `alias`, not `clone` — `Var::clone` mints a fresh TensorId.
            return Ok(input.alias());
        }
        let down_proj = self.mimo_x_down.as_ref().ok_or_else(|| Error::ModelError {
            reason: "Mamba3 mimo_rank > 0 requires mimo_x_down".into(),
        })?;
        let mimo_dim = self.config.headdim * self.config.mimo_rank;
        let flat = var_reshape(input, &[batch * seq_len * self.config.nheads, mimo_dim])
            .map_err(Error::Numr)?;
        let down = down_proj.forward(client, &flat)?;
        var_reshape(
            &down,
            &[batch, seq_len, self.config.nheads, self.config.headdim],
        )
        .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny_mamba3;
    use super::*;
    use crate::model::mamba::mamba3::config::Mamba3Config;
    use crate::nn::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_mimo_up_down_matches_f64_reference() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(2)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_mimo_rank(2)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_use_d(false);
        let mut mamba = tiny_mamba3(config);
        mamba.mimo_x_up = Some(Linear::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5],
                &[4, 2],
                &device,
            )
            .unwrap(),
            None,
            false,
        ));
        mamba.mimo_x_down = Some(Linear::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0],
                &[2, 4],
                &device,
            )
            .unwrap(),
            None,
            false,
        ));

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2], &device).unwrap(),
            false,
        );
        let up = mamba.apply_mimo_up(&client, &x, 1, 1).unwrap();
        let up_data: Vec<f32> = up.tensor().to_vec();
        let up_expected = [1.0f64, 2.0, 3.0, 0.0];
        for (i, (actual, expected)) in up_data.iter().zip(up_expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 1e-6,
                "up idx={i}: actual={actual}, expected={expected}"
            );
        }

        let down = mamba.apply_mimo_down(&client, &up, 1, 1).unwrap();
        let down_data: Vec<f32> = down.tensor().to_vec();
        let down_expected = [1.0f64, -1.0];
        for (i, (actual, expected)) in down_data.iter().zip(down_expected.iter()).enumerate() {
            assert!(
                (*actual as f64 - expected).abs() < 1e-6,
                "down idx={i}: actual={actual}, expected={expected}"
            );
        }
    }
}

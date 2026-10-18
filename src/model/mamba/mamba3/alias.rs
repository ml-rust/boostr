//! [`Mamba3::alias`]: a cheap duplicate that keeps every parameter's `TensorId`.

use super::layer::Mamba3;
use crate::nn::{Conv1d, Linear};
use numr::autograd::Var;
use numr::runtime::Runtime;

impl<R: Runtime> Mamba3<R> {
    /// Cheap duplicate that keeps every parameter's `TensorId`, for capture in a
    /// `'static` activation-checkpointing closure. Each `Var` goes through
    /// [`Var::alias`], not [`Clone`]: a `clone` mints fresh ids and orphans the gradients.
    pub fn alias(&self) -> Self {
        Self {
            config: self.config.clone(),
            in_proj: self.in_proj.alias(),
            out_proj: self.out_proj.alias(),
            lambda_proj: self.lambda_proj.alias(),
            theta_proj: self.theta_proj.as_ref().map(Linear::alias),
            b_bias: self.b_bias.alias(),
            c_bias: self.c_bias.alias(),
            dt_bias: self.dt_bias.as_ref().map(Var::alias),
            a_log: self.a_log.alias(),
            d_param: self.d_param.as_ref().map(Var::alias),
            bc_norm: self.bc_norm.alias(),
            norm: self.norm.alias(),
            conv1d: self.conv1d.as_ref().map(Conv1d::alias),
            mimo_x_up: self.mimo_x_up.as_ref().map(Linear::alias),
            mimo_x_down: self.mimo_x_down.as_ref().map(Linear::alias),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mamba::mamba3::Mamba3Config;
    use crate::nn::{VarBuilder, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;

    /// `alias()` must keep each parameter's `TensorId`. A `clone` mints new ids.
    #[test]
    fn alias_preserves_parameter_ids() {
        let (client, device) = cpu_setup();
        let config = Mamba3Config::new(4)
            .with_nheads(1)
            .with_expand(1)
            .with_d_state(2)
            .with_complex_rope(true)
            .with_mimo_rank(2)
            .with_use_conv(true)
            .with_use_dt_bias(true)
            .with_use_d(true);
        let mut varmap = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let mamba = Mamba3::init(&config, &mut vb, DType::F32, &client, true).unwrap();

        let aliased = mamba.alias();

        let ids = |m: &Mamba3<CpuRuntime>| -> Vec<TensorId> {
            m.parameters().into_iter().map(|(id, _)| id).collect()
        };
        assert_eq!(ids(&mamba).len(), 14);
        assert_eq!(ids(&mamba), ids(&aliased));
    }
}

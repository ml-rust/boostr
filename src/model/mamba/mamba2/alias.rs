//! [`Mamba2::alias`]: a cheap duplicate that keeps every parameter's `TensorId`.

use super::layer::Mamba2;
use crate::nn::RmsNorm;
use numr::autograd::Var;
use numr::runtime::Runtime;

impl<R: Runtime> Mamba2<R> {
    /// Cheap duplicate that keeps every parameter's `TensorId`, for capture in a
    /// `'static` activation-checkpointing closure. Each `Var` goes through
    /// [`Var::alias`], not [`Clone`]: a `clone` mints fresh ids and orphans the gradients.
    pub fn alias(&self) -> Self {
        Self {
            config: self.config.clone(),
            in_proj: self.in_proj.alias(),
            conv1d: self.conv1d.alias(),
            out_proj: self.out_proj.alias(),
            a_log: self.a_log.alias(),
            dt_bias: self.dt_bias.as_ref().map(Var::alias),
            d_param: self.d_param.as_ref().map(Var::alias),
            norm: self.norm.as_ref().map(RmsNorm::alias),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mamba::mamba2::Mamba2Config;
    use crate::nn::{VarBuilder, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::{Tensor, TensorId};

    /// `alias()` must keep each parameter's `TensorId`. A `clone` mints new ids.
    #[test]
    fn alias_preserves_parameter_ids() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(8)
            .with_nheads(1)
            .with_d_state(4)
            .with_expand(2)
            .with_use_dt_bias(true)
            .with_use_d(true);
        let mut varmap = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let mut mamba = Mamba2::init(&config, &mut vb, DType::F32, &client, true).unwrap();
        let norm_weight = Tensor::<CpuRuntime>::ones(&[config.d_inner()], DType::F32, &device);
        mamba.norm = Some(RmsNorm::new(norm_weight.unwrap(), 1e-5, true));

        let aliased = mamba.alias();

        let ids = |m: &Mamba2<CpuRuntime>| -> Vec<TensorId> {
            m.parameters().into_iter().map(|(id, _)| id).collect()
        };
        assert_eq!(ids(&mamba).len(), 7);
        assert_eq!(ids(&mamba), ids(&aliased));
    }
}

//! [`Mamba1::alias`]: a cheap duplicate that keeps every parameter's `TensorId`.

use super::layer::Mamba1;
use numr::autograd::Var;
use numr::runtime::Runtime;

impl<R: Runtime> Mamba1<R> {
    /// Cheap duplicate that keeps every parameter's `TensorId`, for capture in a
    /// `'static` activation-checkpointing closure. Each `Var` goes through
    /// [`Var::alias`], not [`Clone`]: a `clone` mints fresh ids and orphans the gradients.
    pub fn alias(&self) -> Self {
        Self {
            config: self.config.clone(),
            in_proj: self.in_proj.alias(),
            conv1d: self.conv1d.alias(),
            x_proj: self.x_proj.alias(),
            dt_proj: self.dt_proj.alias(),
            out_proj: self.out_proj.alias(),
            a_log: self.a_log.alias(),
            d_param: self.d_param.as_ref().map(Var::alias),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mamba::mamba1::Mamba1Config;
    use crate::nn::{VarBuilder, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;

    /// `alias()` must keep each parameter's `TensorId`. A `clone` mints new ids.
    #[test]
    fn alias_preserves_parameter_ids() {
        let (client, device) = cpu_setup();
        let config = Mamba1Config::new(4)
            .with_expand(1)
            .with_d_state(2)
            .with_use_d(true);
        let mut varmap = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let mamba = Mamba1::init(&config, &mut vb, DType::F32, &client, true).unwrap();

        let aliased = mamba.alias();

        let ids = |m: &Mamba1<CpuRuntime>| -> Vec<TensorId> {
            m.parameters().into_iter().map(|(id, _)| id).collect()
        };
        assert_eq!(ids(&mamba).len(), 7);
        assert_eq!(ids(&mamba), ids(&aliased));
    }
}

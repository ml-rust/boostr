//! AdaGrad state export and import, keyed by parameter name.
//!
//! Keys are `<name>.sum_sq` and `<name>.master`. AdaGrad has no step key.

use super::types::{AdaGrad, ParamState};
use crate::error::Result;
use crate::optimizer::state_dict::{
    MASTER, SUM_SQ, check_matches, check_state_dtype, group_entries, param_name, put, reject_step,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> AdaGrad<R> {
    /// Export the accumulators and master copies, keyed by parameter name.
    pub(super) fn export_state(
        &self,
        names: &HashMap<TensorId, String>,
    ) -> Result<HashMap<String, Tensor<R>>> {
        let mut out = HashMap::new();
        for (&id, state) in &self.state {
            let name = param_name(names, id)?;
            put(&mut out, name, SUM_SQ, &state.acc)?;
            if let Some(master) = &state.master {
                put(&mut out, name, MASTER, master)?;
            }
        }
        Ok(out)
    }

    /// Replace the state with `state`. The optimizer is unchanged after an error.
    pub(super) fn import_state(
        &mut self,
        ids: &HashMap<String, TensorId>,
        state: &HashMap<String, Tensor<R>>,
    ) -> Result<()> {
        reject_step(state, "AdaGrad")?;
        let entries = group_entries(ids, state, &[SUM_SQ], &[MASTER])?;
        let mut loaded = HashMap::with_capacity(entries.len());
        for entry in &entries {
            let acc = entry.require(SUM_SQ)?;
            let acc_key = entry.key(SUM_SQ);
            check_state_dtype(&acc_key, acc)?;
            let master = entry.get(MASTER);
            if let Some(master) = master {
                check_matches(&entry.key(MASTER), master, &acc_key, acc)?;
            }
            loaded.insert(
                entry.id,
                ParamState {
                    acc: acc.clone_deep()?,
                    master: master.map(Tensor::clone_deep).transpose()?,
                },
            );
        }
        self.state = loaded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::AdaGradConfig;
    use super::*;
    use crate::optimizer::state_dict::test_support::{
        Fixture, assert_load_fails, assert_round_trip, deep, run,
    };
    use crate::optimizer::traits::Optimizer;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn make() -> AdaGrad<CpuRuntime> {
        AdaGrad::new(AdaGradConfig {
            initial_accumulator_value: 0.1,
            ..Default::default()
        })
    }

    #[test]
    fn test_adagrad_round_trip_f32() {
        let state = assert_round_trip(make, false);
        assert!(state.contains_key("layer.w.sum_sq"));
        assert!(state.contains_key("bias.sum_sq"));
        assert!(!state.contains_key("__step"));
        assert!(!state.keys().any(|k| k.ends_with(".master")));
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_adagrad_round_trip_bf16_with_master() {
        let state = assert_round_trip(make, true);
        assert_eq!(state["layer.w.master"].dtype(), DType::F32);
        assert_eq!(state["layer.w.sum_sq"].dtype(), DType::F32);
    }

    #[test]
    fn test_adagrad_load_is_strict() {
        let (client, device) = cpu_setup();
        let fx = Fixture::new(&client, &device, false);
        let mut params = deep(&fx.params);
        let mut opt = make();
        run(&mut opt, &client, &fx, &mut params, 0..2);
        let good = opt.state_dict(&fx.names).unwrap();
        let zeros = |shape: &[usize], dtype: DType| {
            Tensor::<CpuRuntime>::zeros(shape, dtype, &device).unwrap()
        };

        let mut unknown_name = good.clone();
        unknown_name.insert("ghost.sum_sq".into(), zeros(&[3], DType::F32));
        assert_load_fails(&make, &fx, &good, &unknown_name, "ghost.sum_sq");

        let mut bad_dtype = good.clone();
        bad_dtype.insert("bias.sum_sq".into(), zeros(&[3], DType::I32));
        assert_load_fails(&make, &fx, &good, &bad_dtype, "bias.sum_sq");

        let mut bad_master = good.clone();
        bad_master.insert("bias.master".into(), zeros(&[2], DType::F32));
        assert_load_fails(&make, &fx, &good, &bad_master, "bias.master");

        let mut partial = good.clone();
        partial.remove("bias.sum_sq");
        partial.insert("bias.master".into(), zeros(&[3], DType::F32));
        assert_load_fails(&make, &fx, &good, &partial, "bias.sum_sq");

        let mut with_step = good.clone();
        with_step.insert("__step".into(), zeros(&[1], DType::F64));
        assert_load_fails(&make, &fx, &good, &with_step, "__step");

        let err = opt.state_dict(&HashMap::new()).err().unwrap().to_string();
        assert!(err.contains("names"), "{err}");
    }
}

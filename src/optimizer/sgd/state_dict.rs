//! Sgd state export and import, keyed by parameter name.
//!
//! Keys are `<name>.momentum_buffer` and `<name>.master`. Sgd has no step key.

use super::types::{ParamState, Sgd};
use crate::error::Result;
use crate::optimizer::state_dict::{
    MASTER, MOMENTUM_BUFFER, check_matches, check_state_dtype, group_entries, param_name, put,
    reject_step,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> Sgd<R> {
    /// Export the velocity buffers and master copies, keyed by parameter name.
    pub(super) fn export_state(
        &self,
        names: &HashMap<TensorId, String>,
    ) -> Result<HashMap<String, Tensor<R>>> {
        let mut out = HashMap::new();
        for (&id, state) in &self.state {
            if state.buf.is_none() && state.master.is_none() {
                continue;
            }
            let name = param_name(names, id)?;
            if let Some(buf) = &state.buf {
                put(&mut out, name, MOMENTUM_BUFFER, buf)?;
            }
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
        reject_step(state, "Sgd")?;
        let entries = group_entries(ids, state, &[], &[MOMENTUM_BUFFER, MASTER])?;
        let mut loaded = HashMap::with_capacity(entries.len());
        for entry in &entries {
            let buf = entry.get(MOMENTUM_BUFFER);
            let master = entry.get(MASTER);
            if let Some(buf) = buf {
                check_state_dtype(&entry.key(MOMENTUM_BUFFER), buf)?;
            }
            if let Some(master) = master {
                match buf {
                    Some(buf) => {
                        check_matches(&entry.key(MASTER), master, &entry.key(MOMENTUM_BUFFER), buf)?
                    }
                    None => check_state_dtype(&entry.key(MASTER), master)?,
                }
            }
            loaded.insert(
                entry.id,
                ParamState {
                    buf: buf.map(Tensor::clone_deep).transpose()?,
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
    use super::super::types::SgdConfig;
    use super::*;
    use crate::optimizer::state_dict::test_support::{
        Fixture, assert_load_fails, assert_round_trip, deep, run,
    };
    use crate::optimizer::traits::Optimizer;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn make() -> Sgd<CpuRuntime> {
        Sgd::new(SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            nesterov: true,
            ..Default::default()
        })
    }

    #[test]
    fn test_sgd_round_trip_f32() {
        let state = assert_round_trip(make, false);
        assert!(state.contains_key("layer.w.momentum_buffer"));
        assert!(state.contains_key("bias.momentum_buffer"));
        assert!(!state.contains_key("__step"));
        assert!(!state.keys().any(|k| k.ends_with(".master")));
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_sgd_round_trip_bf16_with_master() {
        let state = assert_round_trip(make, true);
        assert_eq!(state["layer.w.master"].dtype(), DType::F32);
        assert_eq!(state["layer.w.momentum_buffer"].dtype(), DType::F32);
    }

    #[test]
    fn test_sgd_without_momentum_exports_no_buffer() {
        let make_plain = || Sgd::<CpuRuntime>::new(SgdConfig::default());
        let state = assert_round_trip(make_plain, false);
        assert!(state.is_empty());
    }

    #[test]
    fn test_sgd_load_is_strict() {
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
        unknown_name.insert("ghost.momentum_buffer".into(), zeros(&[3], DType::F32));
        assert_load_fails(&make, &fx, &good, &unknown_name, "ghost.momentum_buffer");

        let mut bad_dtype = good.clone();
        bad_dtype.insert("bias.momentum_buffer".into(), zeros(&[3], DType::I32));
        assert_load_fails(&make, &fx, &good, &bad_dtype, "bias.momentum_buffer");

        let mut bad_master = good.clone();
        bad_master.insert("bias.master".into(), zeros(&[2], DType::F32));
        assert_load_fails(&make, &fx, &good, &bad_master, "bias.master");

        let mut with_step = good.clone();
        with_step.insert("__step".into(), zeros(&[1], DType::F64));
        assert_load_fails(&make, &fx, &good, &with_step, "__step");

        let err = opt.state_dict(&HashMap::new()).err().unwrap().to_string();
        assert!(err.contains("names"), "{err}");
    }
}

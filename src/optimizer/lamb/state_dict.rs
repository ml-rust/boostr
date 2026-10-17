//! Lamb state export and import, keyed by parameter name.
//!
//! Keys are `<name>.exp_avg`, `<name>.exp_avg_sq` and `<name>.master`, plus `__step`.

use super::types::{Lamb, LambState};
use crate::error::Result;
use crate::optimizer::state_dict::{
    EXP_AVG, EXP_AVG_SQ, MASTER, group_entries, param_name, put_moments, put_step, read_step,
    take_moments,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> Lamb<R> {
    /// Export the moments, master copies and step, keyed by parameter name.
    pub(super) fn export_state(
        &self,
        names: &HashMap<TensorId, String>,
    ) -> Result<HashMap<String, Tensor<R>>> {
        let mut out = HashMap::new();
        let mut device = None;
        for (&id, state) in &self.state {
            let name = param_name(names, id)?;
            put_moments(&mut out, name, &state.m, &state.v, state.master.as_ref())?;
            device = Some(state.m.device());
        }
        if let Some(device) = device {
            put_step(&mut out, self.timestep, device)?;
        }
        Ok(out)
    }

    /// Replace the state with `state`. The optimizer is unchanged after an error.
    pub(super) fn import_state(
        &mut self,
        ids: &HashMap<String, TensorId>,
        state: &HashMap<String, Tensor<R>>,
    ) -> Result<()> {
        let entries = group_entries(ids, state, &[EXP_AVG, EXP_AVG_SQ], &[MASTER])?;
        let timestep = read_step(state, !entries.is_empty())?;
        let mut loaded = HashMap::with_capacity(entries.len());
        for entry in &entries {
            let moments = take_moments(entry)?;
            loaded.insert(
                entry.id,
                LambState {
                    m: moments.m,
                    v: moments.v,
                    master: moments.master,
                },
            );
        }
        self.state = loaded;
        self.timestep = timestep;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::LambConfig;
    use super::*;
    use crate::optimizer::state_dict::test_support::{
        assert_moment_strictness, assert_round_trip, assert_step_is,
    };
    use numr::runtime::cpu::CpuRuntime;

    fn make() -> Lamb<CpuRuntime> {
        Lamb::new(LambConfig::default())
    }

    #[test]
    fn test_lamb_round_trip_f32() {
        let state = assert_round_trip(make, false);
        assert_step_is(&state, 3.0);
        assert!(state.contains_key("layer.w.exp_avg"));
        assert!(!state.keys().any(|k| k.ends_with(".master")));
    }

    #[test]
    fn test_lars_round_trip_f32() {
        assert_round_trip(|| Lamb::<CpuRuntime>::new(LambConfig::lars()), false);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_lamb_round_trip_bf16_with_master() {
        let state = assert_round_trip(make, true);
        assert_eq!(state["layer.w.master"].dtype(), DType::F32);
        assert!(!state.contains_key("bias.master"));
    }

    #[test]
    fn test_lamb_load_is_strict() {
        assert_moment_strictness(make);
    }
}

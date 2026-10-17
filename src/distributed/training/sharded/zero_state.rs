//! Optimizer state export and import for the ZeRO stages.
//!
//! Each rank holds state for its owned parameters only. A rank exports and
//! imports the state of the AdamW inside its stage, so every rank saves its own shard.

use super::zero_base::ZeroOptimizerBase;
use crate::error::Result;
use crate::optimizer::Optimizer;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> ZeroOptimizerBase<R> {
    /// Export the state of the owned parameters, keyed by parameter name.
    pub(crate) fn state_dict(
        &self,
        names: &HashMap<TensorId, String>,
    ) -> Result<HashMap<String, Tensor<R>>> {
        Optimizer::state_dict(&self.optimizer, names)
    }

    /// Replace the state of the owned parameters with `state`.
    pub(crate) fn load_state_dict(
        &mut self,
        ids: &HashMap<String, TensorId>,
        state: &HashMap<String, Tensor<R>>,
    ) -> Result<()> {
        Optimizer::load_state_dict(&mut self.optimizer, ids, state)
    }
}

/// Add `state_dict` and `load_state_dict` to a stage type with a `base` field.
macro_rules! impl_zero_state_dict {
    ($ty:ident) => {
        impl<R: numr::runtime::Runtime<DType = numr::dtype::DType>> $ty<R> {
            /// Export the optimizer state of this rank's owned parameters.
            ///
            /// `names` maps parameter ids to names. See
            /// [`Optimizer::state_dict`](crate::optimizer::Optimizer::state_dict) for the keys.
            pub fn state_dict(
                &self,
                names: &std::collections::HashMap<numr::tensor::TensorId, String>,
            ) -> $crate::error::Result<std::collections::HashMap<String, numr::tensor::Tensor<R>>>
            {
                self.base.state_dict(names)
            }

            /// Replace the optimizer state of this rank's owned parameters.
            ///
            /// `ids` maps parameter names to ids. See
            /// [`Optimizer::load_state_dict`](crate::optimizer::Optimizer::load_state_dict) for the checks.
            pub fn load_state_dict(
                &mut self,
                ids: &std::collections::HashMap<String, numr::tensor::TensorId>,
                state: &std::collections::HashMap<String, numr::tensor::Tensor<R>>,
            ) -> $crate::error::Result<()> {
                self.base.load_state_dict(ids, state)
            }
        }
    };
}

pub(crate) use impl_zero_state_dict;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::AdamWConfig;
    use crate::test_utils::cpu_setup;
    use numr::autograd::GradStore;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;
    use std::sync::Arc;

    #[test]
    fn test_base_state_round_trip() {
        let (client, device) = cpu_setup();
        let id = TensorId::new();
        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let g = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
        let comm = Arc::new(NoOpCommunicator);
        let mut base =
            ZeroOptimizerBase::<CpuRuntime>::new(AdamWConfig::default(), comm.clone(), &[id]);
        let mut params = HashMap::new();
        params.insert(id, w);
        let mut grads = GradStore::new();
        grads.insert(id, g);
        base.optimizer.step(&client, &mut params, &grads).unwrap();

        let names = HashMap::from([(id, "w".to_string())]);
        let ids = HashMap::from([("w".to_string(), id)]);
        let state = base.state_dict(&names).unwrap();
        assert!(state.contains_key("w.exp_avg"));

        let mut fresh = ZeroOptimizerBase::<CpuRuntime>::new(AdamWConfig::default(), comm, &[id]);
        fresh.load_state_dict(&ids, &state).unwrap();
        assert_eq!(fresh.optimizer.timestep(), 1);
        assert!(fresh.optimizer.has_state(id));
    }
}

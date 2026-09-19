//! CPU implementation of `ScheduleTuning`.

use crate::quant::QuantFormat;
use crate::quant::traits::ScheduleTuning;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl ScheduleTuning<CpuRuntime> for CpuClient {
    /// No schedule choice is measured on the CPU backend, so this is a
    /// no-op.
    fn warm_schedule_tuning(&self, _formats: &[QuantFormat]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn warm_schedule_tuning_is_callable_and_a_no_op() {
        let (client, _device) = cpu_setup();
        // Nothing to assert on: the call must not panic or change any
        // observable state.
        client.warm_schedule_tuning(&[QuantFormat::Q8_0, QuantFormat::PQ2_0]);
        client.warm_schedule_tuning(&[]);
    }
}

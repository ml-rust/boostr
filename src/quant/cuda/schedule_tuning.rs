//! CUDA implementation of `ScheduleTuning`.

use crate::quant::QuantFormat;
use crate::quant::traits::ScheduleTuning;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl ScheduleTuning<CudaRuntime> for CudaClient {
    fn warm_schedule_tuning(&self, formats: &[QuantFormat]) {
        super::warm_schedule_tuning(self, formats);
    }
}

//! Runtime and fixture setup for unit tests.

use numr::runtime::cpu::{CpuClient, CpuDevice};
#[cfg(feature = "neucodec")]
use std::path::PathBuf;

/// Create a CPU client and device for use in unit tests.
pub(crate) fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// The CUDA client and device for an in-process unit test, plus the
/// guard that serializes every such test. `None` when the feature is on
/// but no device is present, so the test skips instead of failing.
///
/// Serialization is load-bearing, not a convenience. The device client
/// is process-wide with ONE stream. A graph capture on that stream also
/// records whatever another test thread enqueues meanwhile, and a host
/// upload from that thread's freed temporary crashes the replay. Hold
/// the guard for the whole test body.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_setup() -> Option<CudaTest> {
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    use std::sync::{Mutex, OnceLock};

    static SERIAL: OnceLock<Mutex<()>> = OnceLock::new();
    let serial = SERIAL
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !is_cuda_available() {
        eprintln!("CUDA feature enabled but runtime unavailable, skipping");
        return None;
    }
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    Some(CudaTest {
        client,
        device,
        _serial: serial,
    })
}

/// See [`cuda_setup`]. Keep the whole value alive for the test body.
#[cfg(feature = "cuda")]
pub(crate) struct CudaTest {
    pub(crate) client: numr::runtime::cuda::CudaClient,
    pub(crate) device: numr::runtime::cuda::CudaDevice,
    _serial: std::sync::MutexGuard<'static, ()>,
}

/// Resolve the NeuCodec checkpoint fixture: `$NEUCODEC_CHECKPOINT`, else
/// `$BOOSTR_MODELS_DIR/neucodec/model.safetensors`. `None` when neither is
/// set or the resolved path is absent, so callers skip.
#[cfg(feature = "neucodec")]
pub(crate) fn neucodec_checkpoint() -> Option<PathBuf> {
    let path = match std::env::var("NEUCODEC_CHECKPOINT") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("BOOSTR_MODELS_DIR").ok()?)
            .join("neucodec/model.safetensors"),
    };
    path.exists().then_some(path)
}

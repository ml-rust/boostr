//! CUDA kernel loading and caching for boostr attention operations.

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

/// Directory containing the compiled fatbins (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load a compiled fatbin. `Ptx` is cudarc's name for any module image
/// `cuModuleLoadData` accepts: PTX, cubin or fatbin.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.fatbin", KERNEL_DIR, name);
    Ptx::from_file(path)
}

/// Cache for loaded CUDA modules, keyed by (device_index, module_name)
#[allow(clippy::type_complexity)]
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>> =
    OnceLock::new();

/// Get or load a CUDA module from its fatbin.
pub fn get_or_load_module(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_name: &'static str,
) -> Result<Arc<CudaModule>> {
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|e| Error::KernelError {
        reason: format!("kernel cache mutex poisoned: {e}"),
    })?;

    let key = (device_index, module_name);
    if let Some(module) = guard.get(&key) {
        return Ok(module.clone());
    }

    let ptx = load_ptx(module_name);
    let module = context.load_module(ptx).map_err(|e| Error::KernelError {
        reason: format!(
            "Failed to load CUDA module '{}': {:?}. \
             Ensure CUDA kernels were compiled correctly by build.rs.",
            module_name, e
        ),
    })?;

    guard.insert(key, module.clone());
    Ok(module)
}

/// Pre-load a list of CUDA modules to avoid JIT compilation latency on first use.
pub fn preload_modules(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_names: &[&'static str],
) -> Result<()> {
    for name in module_names {
        get_or_load_module(context, device_index, name)?;
    }
    Ok(())
}

/// Get a kernel function from a loaded module.
pub fn get_kernel_function(module: &Arc<CudaModule>, kernel_name: &str) -> Result<CudaFunction> {
    module
        .load_function(kernel_name)
        .map_err(|e| Error::KernelError {
            reason: format!(
                "Failed to get kernel '{}': {:?}. \
                 Check that the kernel name matches the CUDA source.",
                kernel_name, e
            ),
        })
}

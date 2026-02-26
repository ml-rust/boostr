//! CUDA kernel loading for boostr quantized operations

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

/// Directory containing compiled PTX files (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load PTX from compiled file.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.ptx", KERNEL_DIR, name);
    Ptx::from_file(path)
}

/// Module names
pub const DEQUANT_MODULE: &str = "dequant";
pub const QUANT_MATMUL_MODULE: &str = "quant_matmul";
pub const INT4_GEMM_MODULE: &str = "int4_gemm";
pub const INT4_GEMM_GPTQ_MODULE: &str = "int4_gemm_gptq";
pub const NF4_QUANT_MODULE: &str = "nf4_quant";
pub const MARLIN_GEMM_MODULE: &str = "marlin_gemm";
pub const FUSED_INT4_SWIGLU_MODULE: &str = "fused_int4_swiglu";
pub const FUSED_INT4_QKV_MODULE: &str = "fused_int4_qkv";

/// Cache for loaded CUDA modules, keyed by (device_index, module_name)
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>> =
    OnceLock::new();

/// Get or load a CUDA module from PTX.
pub fn get_or_load_module(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_name: &'static str,
) -> Result<Arc<CudaModule>> {
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|e| Error::QuantError {
        reason: format!("kernel cache mutex poisoned: {e}"),
    })?;

    let key = (device_index, module_name);
    if let Some(module) = guard.get(&key) {
        return Ok(module.clone());
    }

    let ptx = load_ptx(module_name);
    let module = context.load_module(ptx).map_err(|e| Error::QuantError {
        reason: format!(
            "Failed to load CUDA module '{}': {:?}. \
                 Ensure CUDA kernels were compiled correctly by build.rs.",
            module_name, e
        ),
    })?;

    guard.insert(key, module.clone());
    Ok(module)
}

/// Get a kernel function from a loaded module.
pub fn get_kernel_function(module: &Arc<CudaModule>, kernel_name: &str) -> Result<CudaFunction> {
    module
        .load_function(kernel_name)
        .map_err(|e| Error::QuantError {
            reason: format!(
                "Failed to get kernel '{}': {:?}. \
                 Check that the kernel name matches the CUDA source.",
                kernel_name, e
            ),
        })
}
